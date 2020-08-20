#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#-*- coding: utf-8 -*-

import parl
from parl import layers
from copy import deepcopy
from paddle import fluid


class DDPG(parl.Algorithm):
    def __init__(self,
                 model,
                 gamma=None,
                 tau=None,
                 actor_lr=None,
                 critic_lr=None):
        """  DDPG algorithm
        
        Args:
            model (parl.Model): actor and critic 的前向网络.
                                model 必须实现 get_actor_params() 方法.
            gamma (float): reward的衰减因子.
            tau (float): self.target_model 跟 self.model 同步参数 的 软更新参数
            actor_lr (float): actor 的学习率
            critic_lr (float): critic 的学习率
        """
        assert isinstance(gamma, float)
        assert isinstance(tau, float)
        assert isinstance(actor_lr, float)
        assert isinstance(critic_lr, float)
        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.model = model
        self.target_model = deepcopy(model)

    def predict(self, obs):
        """ 使用 self.model 的 actor model 来预测动作
        """
        return self.model.policy(obs)

    def learn(self, obs, action, reward, next_obs, terminal):
        """ 用DDPG算法更新 actor 和 critic
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

    def sync_target(self, decay=None, share_vars_parallel_executor=None):
        """ self.target_model从self.model复制参数过来，若decay不为None,则是软更新
        """
        if decay is None:
            decay = 1.0 - self.tau
        self.model.sync_weights_to(
            self.target_model,
            decay=decay,
            share_vars_parallel_executor=share_vars_parallel_executor)
