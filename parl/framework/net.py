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


class Feedforward(Network):
    """
    A feedforward network can contain a sequence of components,
    where each component can be either a LayerFunc or a Feedforward.
    The purpose of this class is to create a collection of LayerFuncs that can
    be easily copied from one Network to another.
    """

    def __init__(self, components):
        for i in range(len(components)):
            setattr(self, "ff%04d" % i, components[i])

    def __call__(self, input):
        attrs = {
            attr: getattr(self, attr)
            for attr in dir(self) if "ff" in attr
        }
        for k in sorted(attrs.keys()):
            input = attrs[k](input)
        return input


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
    1. _predict(): receives a policy state and computes actions
    2. _learn(): receives a policy state and computs a cost

    An algorithm should be part of a network. The user only needs to
    implement the rest of the network in the Model class.
    """

    def __init__(self, model_func, gpu_id):
        super(Algorithm, self).__init__()
        ## When it is called, model_func() will create a new set of model paras
        ## It should be a function with no argument.
        self.set_model_func(model_func)
        self.ref_alg = self
        self.place = fluid.CPUPlace() if gpu_id < 0 \
                     else fluid.CUDAPlace(gpu_id)

    def init(self):
        """
        A callback function inserted right after the algorithm is created,
        but *before* the program is defined.
        See ComputationTask.__init__() for details.
        """
        pass

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

    def set_model_func(self, model_func):
        self.model_func = model_func
        self.model = model_func()
        check_duplicate_spec_names(self.model)

    def predict(self, inputs, states):
        """
        Return a dictionary of results
        """
        policy_states, states = self.model.perceive(inputs,
                                                    states)  # problem-specific
        actions = self._predict(policy_states)  # general algorithm
        actions, states = self.model.post_prediction(
            inputs, actions, states)  # problem-specific

        #### check data specs matching
        specs = dict(self.get_action_specs() + self.get_state_specs())
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
        policy_states, _ = self.model.perceive(inputs,
                                               states)  # problem-specific
        next_policy_states, _ = self.ref_alg.model.perceive(
            next_inputs, next_states)  # problem-specific
        cost = self._learn(policy_states, next_policy_states, actions,
                           rewards)  # general algorithm
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
        Given an internal hidden state, this function predicts actions based on the state.
        The return should be a dictionary containing different kinds of actions.
        Input: policy_states(dict)
        Output: actions(dict)
        """
        raise NotImplementedError()

    def _learn(self, policy_states, next_policy_states, actions, rewards):
        """
        Given an internal hidden state, a dictionary of taken actions, a reward,
        and a next hidden state, this function computes a learning cost to be optimized.
        The return should be the cost.
        Input: policy_states(dict), next_policy_states(dict), actions(dict), rewards(dict)
        Output: cost(dict)
        """
        raise NotImplementedError()


def create_algorithm_func(model_class, model_args, algorithm_class,
                          algorithm_args):
    """
    User API for creating an algorithm lambda function
    Given a model class type, an algorithm class type, and their corresponding
    __init__ args, this function returns an algorithm function that can be called
    without args. Each time the algorithm function is called, it returns a totally
    new algorithm object with its own parameter memory.
    You can pass an algorithm func to create a ComputationTask object.

    The reason we use lambda functions (closure) for both Model and Algorithm is
    that later copies can be created without specifying the args again.
    """
    assert issubclass(model_class, Model)
    assert isinstance(model_args, dict)
    assert issubclass(algorithm_class, Algorithm)
    assert isinstance(algorithm_args, dict)
    model_func = lambda: model_class(**model_args)
    algorithm_func = lambda: algorithm_class(model_func=model_func,
                                             **algorithm_args)
    return algorithm_func
