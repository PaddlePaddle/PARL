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

from parl.framework.net import Algorithm
import parl.layers as layers
from parl.layers import common_functions as comf
from copy import deepcopy


class SimpleAC(Algorithm):
    """
    A simple Actor-Critic that has a feedforward policy network and
    a single discrete action.
    """

    def __init__(self,
                 model,
                 num_actions,
                 mlp_layer_confs,
                 gpu_id=-1,
                 discount_factor=0.99,
                 min_exploration=0.01):

        super(SimpleAC, self).__init__(model, gpu_id)
        self.num_actions = num_actions
        self.discount_factor = discount_factor
        self.min_exploration = min_exploration
        ## create some layers for later use
        self.mlp = [layers.fc(**c) for c in mlp_layer_confs]
        self.policy_layer = layers.fc(num_actions, act='softmax')
        self.value_layer = layers.fc(1)

    def _predict(self, policy_states):
        assert len(policy_states) == 1
        policy = self.policy_layer(
            comf.feedforward(policy_states.values()[0], self.mlp))
        policy = comf.sum_to_one_norm_layer(policy + self.min_exploration)
        return dict(action=comf.discrete_random(policy))

    def _learn(self, policy_states, next_values, actions, rewards):
        assert len(policy_states) == 1
        assert len(actions) == 1
        assert len(next_values) == 1
        assert len(rewards) == 1
        action = actions.values()[0]
        next_value = next_values.values()[0]
        reward = rewards.values()[0]

        critic_value = reward + self.discount_factor * next_value
        td_error = critic_value - self._value(policy_states).values()[0]
        value_cost = layers.square(td_error)

        policy_state = policy_states.values()[0]
        policy = self.policy_layer(comf.feedforward(policy_state, self.mlp))
        policy = comf.sum_to_one_norm_layer(policy + self.min_exploration)
        pg_cost = layers.cross_entropy(input=policy, label=action)
        return dict(cost=value_cost + pg_cost * td_error)

    def _value(self, policy_states):
        assert len(policy_states) == 1
        return dict(value=self.value_layer(
            comf.feedforward(policy_states.values()[0], self.mlp)))


class SimpleQ(Algorithm):
    """
    A simple Q-learning that has a feedforward policy network and a single discrete action.
    """

    def __init__(self,
                 model,
                 num_actions,
                 mlp_layer_confs,
                 gpu_id=-1,
                 discount_factor=0.99,
                 min_exploration=0.01,
                 update_ref_interval=100):

        super(SimpleQ, self).__init__(model, gpu_id)
        self.num_actions = num_actions
        self.discount_factor = discount_factor
        self.min_exploration = min_exploration
        self.gpu_id = gpu_id
        assert update_ref_interval > 0
        self.update_ref_interval = update_ref_interval
        self.total_batches = 0
        ## create some layers for later use
        self.mlp = [layers.fc(**c) for c in mlp_layer_confs]
        ## create a ref alg
        self.ref_alg = deepcopy(self)

    def get_reference_alg(self):
        return self.ref_alg

    def before_every_batch(self):
        if self.total_batches % self.update_ref_interval == 0:
            self.sync_paras_to(self.ref_alg, self.ref_alg.gpu_id)
        self.total_batches += 1

    def _predict(self, policy_states):
        q_values = self._value(policy_states).values()[0]

        max_id = comf.maxid_layer(q_values)
        prob = layers.cast(
            x=layers.one_hot(
                input=max_id, depth=self.num_actions),
            dtype="float32")
        policy = comf.sum_to_one_norm_layer(prob + self.min_exploration)
        return dict(action=comf.discrete_random(policy))

    def _learn(self, policy_states, next_values, actions, rewards):
        assert len(actions) == 1
        assert len(policy_states) == 1
        assert len(next_values) == 1
        assert len(rewards) == 1
        action = actions.values()[0]
        next_q_values = next_values.values()[0]
        reward = rewards.values()[0]

        next_value = layers.reduce_max(next_q_values, dim=-1)

        q_values = self._value(policy_states).values()[0]
        select = layers.cast(
            x=layers.one_hot(
                input=action, depth=self.num_actions),
            dtype="float32")
        value = comf.inner_prod(select, q_values)
        critic_value = reward + self.discount_factor * next_value
        td_error = critic_value - value
        return dict(cost=layers.square(td_error))

    def _value(self, policy_states):
        assert len(policy_states) == 1
        return dict(value=comf.feedforward(policy_states.values()[0],
                                           self.mlp))
