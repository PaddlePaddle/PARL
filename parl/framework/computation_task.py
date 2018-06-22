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
from parl.common.data_process import DataSpecs
from parl.common.utils import split_list
from parl.framework.algorithm import Model, Algorithm
from parl.framework.computation_wrapper import ComputationWrapper


class ComputationTask(object):
    """
    A ComputationTask is responsible for the general data flow
    outside the algorithm

    A ComputationTask is created in a bottom-up way:
    a. create a Model
    b. create an Algorithm with the model as an input
    c. define a ComputationTask with the algorithm
    """

    def __init__(self, name, algorithm, **kwargs):
        assert isinstance(algorithm, Algorithm)
        self.name = name
        self.alg = algorithm
        ## create an Fluid executor
        self._define_program()
        place = fluid.CPUPlace() if self.alg.gpu_id < 0 \
                else fluid.CUDAPlace(self.alg.gpu_id)
        self.fluid_executor = fluid.Executor(place)
        self.fluid_executor.run(fluid.default_startup_program())
        self._wrapper_args = kwargs
        self._wrapper = None

    @property
    def specs(self):
        assert self._specs, ("_specs should've been defined after __init__.")
        return self._specs

    @property
    def wrapper(self):
        if self._wrapper is None:
            self._wrapper = ComputationWrapper(self.name, self,
                                               **self._wrapper_args)
        return self._wrapper

    def _create_data_layers(self, specs):
        data_layers = {}
        for name, args in specs:
            data_layers[name] = layers.data(name, **args)
        return data_layers

    def _define_program(self):
        self.learn_program = fluid.Program()
        self.predict_program = fluid.Program()

        def _select_data(data_layer_dict, specs):
            return {name: data_layer_dict[name] for name, _ in specs}

        self._specs = DataSpecs(
            inputs=self.alg.get_input_specs(),
            states=self.alg.get_state_specs(),
            actions=self.alg.get_action_specs(),
            rewards=self.alg.get_reward_specs(),
            next_episode_end=[("next_episode_end", dict(shape=[1]))])

        self.action_names = sorted([name for name, _ in self._specs.actions])
        self.state_names = sorted([name for name, _ in self._specs.states])

        with fluid.program_guard(self.predict_program):
            data_layer_dict = self._create_data_layers(self._specs.inputs)
            data_layer_dict.update(
                self._create_data_layers(self._specs.states))
            self.predict_feed_names = sorted(data_layer_dict.keys())

            inputs = _select_data(data_layer_dict, self._specs.inputs)
            states = _select_data(data_layer_dict, self._specs.states)

            ### call alg predict()
            pred_actions, pred_states = self.alg.predict(inputs, states)
            self.predict_fetch = [pred_actions, pred_states]

        with fluid.program_guard(self.learn_program):
            data_layer_dict = self._create_data_layers(self._specs.inputs)
            data_layer_dict.update(
                self._create_data_layers(self._specs.states))
            data_layer_dict.update(
                self._create_data_layers(self._specs.next_inputs))
            data_layer_dict.update(
                self._create_data_layers(self._specs.next_states))
            data_layer_dict.update(
                self._create_data_layers(self._specs.actions))
            data_layer_dict.update(
                self._create_data_layers(self._specs.rewards))
            data_layer_dict.update(
                self._create_data_layers(self._specs.next_episode_end))
            self.learn_feed_names = sorted(data_layer_dict.keys())

            inputs = _select_data(data_layer_dict, self._specs.inputs)
            states = _select_data(data_layer_dict, self._specs.states)
            next_inputs = _select_data(data_layer_dict,
                                       self._specs.next_inputs)
            next_states = _select_data(data_layer_dict,
                                       self._specs.next_states)
            actions = _select_data(data_layer_dict, self._specs.actions)
            rewards = _select_data(data_layer_dict, self._specs.rewards)
            next_episode_end = _select_data(data_layer_dict,
                                            self._specs.next_episode_end)

            ## call alg learn()
            ### TODO: implement a recurrent layer to strip the sequence information
            ### TODO: algorithm.learn returns customized outputs not a fixed cost
            self.cost = self.alg.learn(inputs, next_inputs, states,
                                       next_states, next_episode_end, actions,
                                       rewards)

    def predict(self, inputs, states=dict()):
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
        assert sorted(actions.keys()) == self.action_names
        assert sorted(states.keys()) == self.state_names
        return actions, states

    def learn(self,
              inputs,
              next_inputs,
              next_episode_end,
              actions,
              rewards,
              states=dict(),
              next_states=dict()):
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
        data.update(next_episode_end)
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
