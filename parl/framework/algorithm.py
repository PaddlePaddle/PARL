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
import parl.framework.policy_distribution as pd


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
    A Model is owned by an Algorithm. It implements all the network model of
    a specific problem.
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

    def policy(self, inputs, states):
        """
        Return: action_dists: a dict of action distribution objects
                states
                An action distribution object can be created with
                PolicyDistribution().
        Optional: a model might not always have to implement policy()
        """
        raise NotImplementedError()

    def value(self, inputs, states):
        """
        Return: values: a dict of estimated values for the current observations and states
                        For example, "q_value" and "v_value"
        Optional: a model might not always have to implement value()
        """
        raise NotImplementedError()


class Algorithm(object):
    """
    An Algorithm implements two functions:
    1. predict() computes forward
    2. learn() computes a cost for optimization

    An algorithm should be only part of a network. The user only needs to
    implement the rest of the network in the Model class.
    """

    def __init__(self, model, hyperparas, gpu_id):
        assert isinstance(model, Model)
        check_duplicate_spec_names(model)
        self.model = model
        self.hp = hyperparas
        self.gpu_id = gpu_id

    def get_input_specs(self):
        return self.model.get_input_specs()

    def get_state_specs(self):
        return self.model.get_state_specs()

    def get_action_specs(self):
        """
        For non-RL algortihms, this can return []
        """
        return self.model.get_action_specs()

    def get_reward_specs(self):
        """
        For non-RL algortihms, this can return []
        """
        return self.model.get_reward_specs()

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
        Given the inputs and states, this function does forward prediction and updates states.
        """
        raise NotImplementedError()

    def learn(self, inputs, next_inputs, states, next_states, episode_end,
              actions, rewards):
        """
        This function computes a learning cost to be optimized.
        The return should be the cost.
        Output: cost(dict)
        """
        pass


class RLAlgorithm(Algorithm):
    """
    A derived Algorithm class specially for RL problems.
    """

    def __init__(self, model, hyperparas, gpu_id):
        super(RLAlgorithm, self).__init__(model, hyperparas, gpu_id)

    def get_behavior_model(self):
        """
        Return the behavior model to compute actions. The behavior model could be different
        from the training model, which is common in off-policy RL algorithms.

        The default behavior model is set to the training model. The user can override this
        function to specify another different model.
        """
        return self.model

    def predict(self, inputs, states):
        """
        Implementation of Algorithm.predict()

        Given the inputs and states, this function predicts actions and updates states.
        Input: inputs(dict), states(dict)
        Output: actions(dict), states(dict)
        """
        behavior_model = self.get_behavior_model()
        distributions, states = behavior_model.policy(inputs, states)
        actions = {}
        for key, dist in distributions.iteritems():
            assert isinstance(
                dist, pd.PolicyDistribution
            ), "behavior_model.policy must return PolicyDist!"
            actions[key] = dist()
        return actions, states
