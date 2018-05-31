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

import paddle.fluid as fluid
import parl.layers as layers
from parl.framework.net import Model, Algorithm


def split_list(l, sizes):
    """
    Split a list into several chunks, each chunk with a size in sizes
    """
    chunks = []
    offset = 0
    for size in sizes:
        chunks.append(l[offset:offset + size])
        offset += size
    return chunks


class ComputationTask(object):
    """
    A ComputationTask is responsible for the general data flow
    outside the algorithm

    A ComputationTask is created in a bottom-up way:
    a. create a Model
    b. create an Algorithm with the model as an input
    c. define a ComputationTask with the algorithm
    """

    def __init__(self, algorithm):
        assert isinstance(algorithm, Algorithm)
        self.alg = algorithm
        ## create an Fluid executor
        self._define_program()
        place = fluid.CPUPlace() if self.alg.gpu_id < 0 \
                else fluid.CUDAPlace(self.alg.gpu_id)
        self.fluid_executor = fluid.Executor(place)
        self.fluid_executor.run(fluid.default_startup_program())

    def _create_data_layers(self, specs):
        data_layers = {}
        for name, args in specs:
            data_layers[name] = layers.data(name, **args)
        return data_layers

    def _define_program(self):
        self.learn_program = fluid.Program()

        def _get_next_specs(specs):
            return [("next_" + spec[0], spec[1]) for spec in specs]

        def _select_data(data_layer_dict, specs):
            return {name: data_layer_dict[name] for name, _ in specs}

        input_specs = self.alg.get_input_specs()
        state_specs = self.alg.get_state_specs()
        next_input_specs = _get_next_specs(input_specs)
        next_state_specs = _get_next_specs(state_specs)
        action_specs = self.alg.get_action_specs()
        reward_specs = self.alg.get_reward_specs()

        with fluid.program_guard(self.learn_program):
            data_layer_dict = self._create_data_layers(input_specs)
            data_layer_dict.update(self._create_data_layers(state_specs))
            self.predict_feed_names = sorted(data_layer_dict.keys())

            data_layer_dict.update(self._create_data_layers(next_input_specs))
            data_layer_dict.update(self._create_data_layers(next_state_specs))
            data_layer_dict.update(self._create_data_layers(action_specs))
            data_layer_dict.update(self._create_data_layers(reward_specs))
            self.learn_feed_names = sorted(data_layer_dict.keys())

            inputs = _select_data(data_layer_dict, input_specs)
            states = _select_data(data_layer_dict, state_specs)
            next_inputs = _select_data(data_layer_dict, next_input_specs)
            next_states = _select_data(data_layer_dict, next_state_specs)
            actions = _select_data(data_layer_dict, action_specs)
            rewards = _select_data(data_layer_dict, reward_specs)

            ### call alg predict()
            pred_actions, pred_states = self.alg.predict(inputs, states)
            self.predict_fetch = [pred_actions, pred_states]

            ## up to this point is the predict program
            self.predict_program = self.learn_program.clone()

            ## call alg learn()
            ### TODO: implement a recurrent layer to strip the sequence information
            self.cost = self.alg.learn(inputs, next_inputs, states,
                                       next_states, actions, rewards)

    def predict(self, inputs, states):
        """
        ComputationTask predict API
        This function is responsible to convert Python data to Fluid tensors, and
        then convert the computational results in the reverse way.
        """
        data = {}
        data.update(inputs)
        data.update(states)
        assert sorted(data.keys()) == self.predict_feed_names, \
            "field names mismatch: %s %s" % (data.keys(), self.predict_feed_names)
        feed = {n: data[n] for n in self.predict_feed_names}

        ### run the predict_program and fetch the computational results
        action_tensors, state_tensors = self.predict_fetch
        action_tensors = list(action_tensors.iteritems())
        state_tensors = list(state_tensors.iteritems())
        result = self.fluid_executor.run(
            self.predict_program,
            feed=feed,
            fetch_list=[t for _, t in action_tensors + state_tensors])

        ## actions and states are numpy arrays
        actions, states = split_list(
            result, [len(action_tensors), len(state_tensors)])

        ## wrap the results into dictionaries for better access
        actions = dict(zip([name for name, _ in action_tensors], actions))
        states = dict(zip([name for name, _ in state_tensors], states))
        return actions, states

    def learn(self, inputs, next_inputs, states, next_states, actions,
              rewards):
        """
        ComputationTask learn API
        This function is responsible to convert Python data to Fluid tensors, and
        then convert the computational results in the reverse way.
        """
        data = {}
        data.update(inputs)
        data.update(next_inputs)
        data.update(states)
        data.update(next_states)
        data.update(actions)
        data.update(rewards)
        assert sorted(data.keys()) == self.learn_feed_names, \
            "field names mismatch: %s %s" % ()
        feed = {n: data[n] for n in self.learn_feed_names}

        self.alg.before_every_batch()
        ## run the learn program and fetch the sole cost output
        result = self.fluid_executor.run(self.learn_program,
                                         feed=feed,
                                         fetch_list=[self.cost["cost"]])
        self.alg.after_every_batch()
        return dict(cost=result[0])
