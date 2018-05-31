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
from parl.layers import Network
from paddle.fluid.framework import convert_np_dtype_to_dtype_
import inspect


def check_duplicate_spec_names(model):
    """
    Check if there are two specs that have the same name.
    """
    specs = model.get_input_specs() \
            + model.get_action_specs() \
            + model.get_state_specs() \
            + model.get_reward_specs()
    names = [name for name, _ in specs]
    duplicates = set([n for n in names if names.count(n) > 1])
    assert not duplicates, \
        "duplicate names with different specs: " + " ".join(duplicates)


class Model(Network):
    """
    A Model is owned by an Algorithm . It implements four functions:
    1. get_input_specs(): defines the input specs
    2. get_state_specs(): defines the state specs
    3. get_action_specs(): defines the action specs
    4. get_reward_specs(): defines the reward specs
    """

    def __init__(self):
        super(Model, self).__init__()

    def get_input_specs(self):
        """
        Output: list of tuples
        """
        raise NotImplementedError()

    def get_state_specs(self):
        """
        States are optional to a Model.
        Output: list of tuples
        """
        return []

    def get_action_specs(self):
        """
        Output: list of tuples
        """
        raise NotImplementedError()

    def get_reward_specs(self):
        """
        By default, a scalar reward.
        User can specify a vector of rewards for some problems
        """
        return [("reward", dict(shape=[1]))]

    def perceive(self, inputs, states):
        """
        Input: inputs(dict), states(dict)
        Ouput: policy_states(dict), states(dict)
        """
        raise NotImplementedError()

    def post_prediction(self, inputs, actions, states):
        """
        Some computations to perform after the actions are predicted.
        By default this function does nothing.

        Input: inputs(dict), actions(dict), states(dict)
        Output: actions(dict), states(dict)
        """
        return actions, states


class Algorithm(Network):
    """
    An Algorithm implements two functions:
    1. _predict() computes actions
    2. _learn() computes a cost for optimization
    3. _value() computes values

    An algorithm should be only part of a network. The user only needs to
    implement the rest of the network in the Model class.
    """

    def __init__(self, model, gpu_id=-1):
        super(Algorithm, self).__init__()
        assert isinstance(model, Model)
        check_duplicate_spec_names(model)
        self.model = model
        self.gpu_id = gpu_id

    def get_reference_alg(self):
        """
        Get the reference algorithm which is used for computing the policy or
        value for next_inputs
        """
        return self

    def get_behavior_alg(self):
        """
        Get the behavior algorithm which is used for computing the prediction
        """
        return self

    def before_every_batch(self):
        """
        A callback function inserted before every batch of training.
        See ComputationTask.learn()
        """
        pass

    def after_every_batch(self):
        """
        A callback function inserted after every batch of training.
        See ComputationTask.learn()
        """
        pass

    def predict(self, inputs, states):
        """
        Return a dictionary of results
        """
        behavior_alg = self.get_behavior_alg()
        policy_states, states = behavior_alg.model.perceive(
            inputs, states)  # problem-specific
        actions = behavior_alg._predict(policy_states)  # general algorithm
        if actions is None:
            return NotImplementedError("behavior_alg._predict not implemented"), \
                NotImplementedError()

        actions, states = behavior_alg.model.post_prediction(
            inputs, actions, states)  # problem-specific

        #### check data specs matching
        specs = dict(behavior_alg.get_action_specs() \
                     + behavior_alg.get_state_specs())
        assert len(actions.keys() + states.keys()) == len(specs)
        for key in actions.keys() + states.keys():
            act = actions[key]
            act_spec = specs[key]
            if "dtype" in act_spec:
                assert act.dtype == convert_np_dtype_to_dtype_(act_spec[
                    "dtype"])
            else:
                ## the default type is none specified
                assert act.dtype == convert_np_dtype_to_dtype_("float32")
        return actions, states

    def learn(self, inputs, next_inputs, states, next_states, actions,
              rewards):
        ref_alg = self.get_reference_alg()
        policy_states, _ = self.model.perceive(inputs,
                                               states)  # problem-specific
        next_policy_states, _ = ref_alg.model.perceive(
            next_inputs, next_states)  # problem-specific
        next_values = ref_alg._value(next_policy_states)
        if next_values is None:
            return NotImplementedError("ref_alg._value() not implemented")

        ## we stop the gradients for all the next values
        for nv in next_values.values():
            nv.stop_gradient = True

        cost = self._learn(policy_states, next_values, actions,
                           rewards)  # general algorithm
        if cost is None:
            return NotImplementedError("_learn() not implemented")

        assert len(cost) == 1 and "cost" in cost
        assert cost["cost"].dtype == convert_np_dtype_to_dtype_("float32") \
            and list(cost["cost"].shape[1:]) == [1]

        avg_cost = layers.mean(x=cost["cost"])
        ## TODO: add customized options for the optimizer
        optimizer = fluid.optimizer.SGD(learning_rate=1e-2)
        optimizer.minimize(avg_cost)
        return dict(cost=avg_cost)

    def get_input_specs(self):
        return self.model.get_input_specs()

    def get_state_specs(self):
        return self.model.get_state_specs()

    def get_action_specs(self):
        return self.model.get_action_specs()

    def get_reward_specs(self):
        return self.model.get_reward_specs()

    def _predict(self, policy_states):
        """
        Given the policy states, this function predicts actions.
        The return should be a dictionary containing different kinds of actions.
        Input: policy_states(dict)
        Output: actions(dict)
        """
        pass

    def _learn(self, policy_states, next_values, actions, rewards):
        """
        Given the policy states, the values at next time steps,
        a dictionary of taken actions, and rewards,
        this function computes a learning cost to be optimized.
        The return should be the cost.
        Input: policy_states(dict), next_policy_states(dict), actions(dict), rewards(dict)
        Output: cost(dict)
        """
        pass

    def _value(self, policy_states):
        """
        Given the policy states, this function computes values for _learn()
        Input: policy_states(dict)
        Output: values(dict)
        """
        pass
