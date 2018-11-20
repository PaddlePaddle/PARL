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

from parl.framework.algorithm import Algorithm
from paddle.fluid.initializer import ConstantInitializer
import parl.layers as layers
import parl.framework.policy_distribution as pd
from parl.layers import common_functions as comf
import paddle.fluid as fluid
from copy import deepcopy


class SimpleAC(Algorithm):
    """
    A simple Actor-Critic that has a feedforward policy network and
    a single discrete action.

    learn() requires keywords: "action", "reward", "v_value"
    """

    def __init__(self,
                 model,
                 hyperparas=dict(lr=1e-4),
                 gpu_id=-1,
                 discount_factor=0.99):

        super(SimpleAC, self).__init__(model, hyperparas, gpu_id)
        self.discount_factor = discount_factor

    def learn(self, inputs, next_inputs, states, next_states, next_episode_end,
              actions, rewards):

        action = actions["action"]
        reward = rewards["reward"]

        values = self.model.value(inputs, states)
        next_values = self.model.value(next_inputs, next_states)
        value = values["v_value"]
        next_value = next_values["v_value"] * next_episode_end[
            "next_episode_end"]
        next_value.stop_gradient = True
        assert value.shape[1] == next_value.shape[1]

        critic_value = reward + self.discount_factor * next_value
        td_error = critic_value - value
        value_cost = layers.square(td_error)

        dist, _ = self.model.policy(inputs, states)
        dist = dist["action"]
        assert isinstance(dist, pd.CategoricalDistribution)

        pg_cost = 0 - dist.loglikelihood(action)
        avg_cost = layers.mean(x=value_cost + pg_cost * td_error)
        optimizer = fluid.optimizer.DecayedAdagradOptimizer(
            learning_rate=self.hp["lr"])
        optimizer.minimize(avg_cost)
        return dict(cost=avg_cost)

    def predict(self, inputs, states):
        return self._rl_predict(self.model, inputs, states)


class SimpleQ(Algorithm):
    """
    A simple Q-learning that has a feedforward policy network and a single discrete action.

    learn() requires keywords: "action", "reward", "q_value"
    """

    def __init__(self,
                 model,
                 hyperparas=dict(lr=1e-4),
                 gpu_id=-1,
                 discount_factor=0.99,
                 exploration_end_batches=0,
                 exploration_end_rate=0.1,
                 update_ref_interval=100):

        super(SimpleQ, self).__init__(model, hyperparas, gpu_id)
        self.discount_factor = discount_factor
        self.gpu_id = gpu_id
        assert update_ref_interval > 0
        self.update_ref_interval = update_ref_interval
        self.total_batches = 0
        ## create a reference model
        self.ref_model = deepcopy(model)
        ## setup exploration
        self.explore = (exploration_end_batches > 0)
        if self.explore:
            self.exploration_counter = layers.create_persistable_variable(
                dtype="float32",
                shape=[1],
                is_bias=True,
                default_initializer=ConstantInitializer(0.))
            ### in the second half of training time, the rate is fixed to a number
            self.total_exploration_batches = exploration_end_batches
            self.exploration_rate_delta \
                = (1 - exploration_end_rate) / self.total_exploration_batches

    def before_every_batch(self):
        if self.total_batches % self.update_ref_interval == 0:
            self.model.sync_paras_to(self.ref_model, self.gpu_id)
        self.total_batches += 1

    def predict(self, inputs, states):
        """
        Override the base predict() function to put the exploration rate in inputs
        """
        rate = 0
        if self.explore:
            counter = self.exploration_counter()
            ## first compute the current exploration rate
            rate = 1 - counter * self.exploration_rate_delta

        distributions, states = self.model.policy(inputs, states)
        for dist in distributions.values():
            assert dist.__class__.__name__ == "CategoricalDistribution"
            dist.add_uniform_exploration(rate)

        actions = {}
        for key, dist in distributions.iteritems():
            actions[key] = dist()
        return actions, states

    def learn(self, inputs, next_inputs, states, next_states, next_episode_end,
              actions, rewards):

        action = actions["action"]
        reward = rewards["reward"]

        values = self.model.value(inputs, states)
        next_values = self.ref_model.value(next_inputs, next_states)
        q_value = values["q_value"]
        next_q_value = next_values["q_value"] * next_episode_end[
            "next_episode_end"]
        next_q_value.stop_gradient = True
        next_value = layers.reduce_max(next_q_value, dim=-1)
        assert q_value.shape[1] == next_q_value.shape[1]
        num_actions = q_value.shape[1]

        value = comf.idx_select(input=q_value, idx=action)
        critic_value = reward + self.discount_factor * next_value
        td_error = critic_value - value

        avg_cost = layers.mean(x=layers.square(td_error))
        optimizer = fluid.optimizer.DecayedAdagradOptimizer(
            learning_rate=self.hp["lr"])
        optimizer.minimize(avg_cost)

        self._increment_exploration_counter()
        return dict(cost=avg_cost)

    def _increment_exploration_counter(self):
        if self.explore:
            counter = self.exploration_counter()
            exploration_counter_ = counter + 1
            switch = layers.cast(
                x=(exploration_counter_ > self.total_exploration_batches),
                dtype="float32")
            ## if the counter already hits the limit, we do not change the counter
            layers.assign(
                switch * counter + (1 - switch) * exploration_counter_,
                counter)
