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

import parl.layers as layers
from copy import deepcopy
from paddle import fluid
from parl.framework.algorithm_base import Algorithm

__all__ = ['DDPG']


class DDPG(Algorithm):
    def __init__(self, model, hyperparas):
        """ model: should implement the function get_actor_params()
        """
        Algorithm.__init__(self, model, hyperparas)
        self.model = model
        self.target_model = deepcopy(model)

        # fetch hyper parameters
        self.gamma = hyperparas['gamma']
        self.tau = hyperparas['tau']
        self.actor_lr = hyperparas['actor_lr']
        self.critic_lr = hyperparas['critic_lr']

    def define_predict(self, obs):
        """ use actor model of self.model to predict the action
        """
        return self.model.policy(obs)

    def define_learn(self, obs, action, reward, next_obs, terminal):
        """ update actor and critic model with DDPG algorithm
        """
        actor_cost = self._actor_learn(obs)
        critic_cost = self._critic_learn(obs, action, reward, next_obs,
                                         terminal)
        return actor_cost, critic_cost

    def _actor_learn(self, obs):
        action = self.model.policy(obs)
        Q = self.model.value(obs, action)
        cost = layers.reduce_mean(-1.0 * Q)
        optimizer = fluid.optimizer.AdamOptimizer(self.actor_lr)
        optimizer.minimize(cost, parameter_list=self.model.get_actor_params())
        return cost

    def _critic_learn(self, obs, action, reward, next_obs, terminal):
        next_action = self.target_model.policy(next_obs)
        next_Q = self.target_model.value(next_obs, next_action)

        terminal = layers.cast(terminal, dtype='float32')
        target_Q = reward + (1.0 - terminal) * self.gamma * next_Q
        target_Q.stop_gradient = True

        Q = self.model.value(obs, action)
        cost = layers.square_error_cost(Q, target_Q)
        cost = layers.reduce_mean(cost)
        optimizer = fluid.optimizer.AdamOptimizer(self.critic_lr)
        optimizer.minimize(cost)
        return cost

    def sync_target(self, gpu_id, decay=None):
        if decay is None:
            decay = 1.0 - self.tau
        self.model.sync_params_to(
            self.target_model, gpu_id=gpu_id, decay=decay)
