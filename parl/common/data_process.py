#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np


class DataSpecs(object):
    """
    `DataSpecs` stores the specs of input data for `ComputationTask`. Its 
    responsibility is to reformat the input data specs returned from `Algorithm`
    so that it is easy for users to parse.

    Currently, the only job of `DataSpecs` is to ensure the validity of specs. 
    In the future, we can implement more complicated data processing routine. 
    """

    def __init__(self,
                 inputs=[],
                 states=[],
                 actions=[],
                 rewards=[],
                 next_episode_end=[]):
        self.inputs = self.__check_specs(inputs)
        self.next_inputs = self.__get_next_specs(self.inputs)
        self.states = self.__check_specs(states)
        self.next_states = self.__get_next_specs(self.states)
        self.actions = self.__check_specs(actions)
        self.rewards = self.__check_specs(rewards)
        self.next_episode_end = self.__check_specs(next_episode_end)

    def __check_specs(self, specs_list):
        """
        Make sure specs are in the correct format.
        """

        assert isinstance(specs_list, list)
        for specs in specs_list:
            assert isinstance(specs, tuple) and len(specs) == 2
            assert isinstance(specs[0], str) and isinstance(specs[1], dict)
            assert "shape" in specs[1] and isinstance(specs[1]["shape"], list)
            if not specs[1]["shape"]:
                # if no `shape` is given, we don't want `dtype`
                specs[1].pop("dtype", None)
            elif not "dtype" in specs[1]:
                specs[1]["dtype"] = "float32"
        return specs_list

    def __get_next_specs(self, specs):
        """
        Generate `next_*` data specs from the existing data specs.
        """

        return [("next_" + spec[0], spec[1]) for spec in specs]


class DataProcessor(object):
    """
    `DataProcessor` processes the prediction/learning data into the format
    specified by a given `specs`.

    Users can provide `process_funcs`, a dictionary of data processing functions,
    to tell `DataProcessor` how to process different kinds of input data, or 
    leave them to default functions.
    """

    def __init__(self, data_specs, process_funcs=dict()):
        assert isinstance(data_specs, DataSpecs)
        assert isinstance(process_funcs, dict)
        self.data_specs = data_specs
        # default data processing functions
        self.process_funcs = dict(
            inputs=self.__prepare_data,
            next_inputs=self.__prepare_data,
            next_episode_end=self.__prepare_data,
            actions=self.__prepare_data,
            rewards=self.__prepare_data,
            states=self.__prepare_state_data,
            next_states=self.__prepare_state_data)
        # overwrite with user-specified functions
        self.process_funcs.update(process_funcs)

    def __assemble_data(self, E, spec):
        """
        Assemble a list of `Experience` into a batch, according to the given
        `spec`. 

        Args:
            E(list of `Experience`): a list of `Experience`.
            spec(tuple): data assembling specs. The `spec` specifies what kind 
            of the experience data to assemble (`spec[0]`) and the format, e.g.,
            shape and data type, of the resulting batch (`spec[1]`).

        Return:
            a list of data or a numpy array representing the resulting batch.
        """

        attr = spec[0]
        if attr.startswith("next_"):
            # data comes from the experience of the next time step
            L = [getattr(e.next_exp, attr[5:]) for e in E]
        else:
            L = [getattr(e, attr) for e in E]
        if not spec[1]["shape"]:
            # no shape specified means the data has variable shapes and we
            # should just return the list
            return L
        else:
            return np.array(L).astype(spec[1]["dtype"])

    def __prepare_data(self, exp_seqs, specs):
        # flatten the experience sequence into one list
        E = [e for exp_seq in exp_seqs for e in exp_seq]
        data = {spec[0]: self.__assemble_data(E, spec) for spec in specs}
        return data

    def __prepare_state_data(self, exp_seqs, specs):
        # For `state` data, only the states of the first `Experience` from each
        # sequence is used.
        E = [exp_seq[0] for exp_seq in exp_seqs]
        data = {spec[0]: self.__assemble_data(E, spec) for spec in specs}
        return data

    def process_prediction_data(self, inputs, states):
        def __convert_to_batch(data, specs):
            new_data = {}
            if data:
                for spec in specs:
                    k = spec[0]
                    # we put `[]` around `data[k]` so that `np.array` will 
                    # add an axis for the dimension of batch size
                    new_data[k] = np.array([data[k]]).astype(spec[1]["dtype"])
            return new_data

        isinstance(inputs, dict)
        isinstance(states, dict)
        data = {"inputs": __convert_to_batch(inputs, self.data_specs.inputs)}
        if self.data_specs.states:
            data["states"] = __convert_to_batch(states, self.data_specs.states)
        return data

    def process_learning_data(self, exp_seqs):
        data = {
            k: self.process_funcs[k](exp_seqs, specs)
            for k, specs in vars(self.data_specs).iteritems()
        }
        return data
