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

from parl.core.fluid import layers
from copy import deepcopy
import numpy as np
from paddle import fluid
from paddle.fluid.layers import Normal
from parl.core.fluid.algorithm import Algorithm

epsilon = 1e-6

__all__ = ['SAC']


class SAC(Algorithm):
    def __init__(self,
                 actor,
                 critic,
                 max_action,
                 alpha=0.2,
                 gamma=None,
                 tau=None,
                 actor_lr=None,
                 critic_lr=None):
        """ SAC algorithm

        Args:
            actor (parl.Model): forward network of actor.
            critic (patl.Model): forward network of the critic.
            max_action (float): the largest value that an action can be, env.action_space.high[0]
            alpha (float): Temperature parameter determines the relative importance of the entropy against the reward
            gamma (float): discounted factor for reward computation.
            tau (float): decay coefficient when updating the weights of self.target_model with self.model
            actor_lr (float): learning rate of the actor model
            critic_lr (float): learning rate of the critic model
        """
        assert isinstance(gamma, float)
        assert isinstance(tau, float)
        assert isinstance(actor_lr, float)
        assert isinstance(critic_lr, float)
        assert isinstance(alpha, float)
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.alpha = alpha

        self.actor = actor
        self.critic = critic
        self.target_critic = deepcopy(critic)

    def predict(self, obs):
        """ use actor model of self.policy to predict the action
        """
        mean, _ = self.actor.policy(obs)
        mean = layers.tanh(mean) * self.max_action
        return mean

    def sample(self, obs):
        mean, log_std = self.actor.policy(obs)
        std = layers.exp(log_std)
        normal = Normal(mean, std)
        x_t = normal.sample([1])[0]
        y_t = layers.tanh(x_t)
        action = y_t * self.max_action
        log_prob = normal.log_prob(x_t)
        log_prob -= layers.log(self.max_action * (1 - layers.pow(y_t, 2)) +
                               epsilon)
        log_prob = layers.reduce_sum(log_prob, dim=1, keep_dim=True)
        log_prob = layers.squeeze(log_prob, axes=[1])
        return action, log_prob

    def learn(self, obs, action, reward, next_obs, terminal):
        actor_cost = self.actor_learn(obs)
        critic_cost = self.critic_learn(obs, action, reward, next_obs,
                                        terminal)
        return critic_cost, actor_cost

    def actor_learn(self, obs):
        action, log_pi = self.sample(obs)
        qf1_pi, qf2_pi = self.critic.value(obs, action)
        min_qf_pi = layers.elementwise_min(qf1_pi, qf2_pi)
        cost = log_pi * self.alpha - min_qf_pi
        cost = layers.reduce_mean(cost)
        optimizer = fluid.optimizer.AdamOptimizer(self.actor_lr)
        optimizer.minimize(cost, parameter_list=self.actor.parameters())

        return cost

    def critic_learn(self, obs, action, reward, next_obs, terminal):
        next_obs_action, next_obs_log_pi = self.sample(next_obs)
        qf1_next_target, qf2_next_target = self.target_critic.value(
            next_obs, next_obs_action)
        min_qf_next_target = layers.elementwise_min(
            qf1_next_target, qf2_next_target) - next_obs_log_pi * self.alpha

        terminal = layers.cast(terminal, dtype='float32')
        target_Q = reward + (1.0 - terminal) * self.gamma * min_qf_next_target
        target_Q.stop_gradient = True

        current_Q1, current_Q2 = self.critic.value(obs, action)
        cost = layers.square_error_cost(current_Q1,
                                        target_Q) + layers.square_error_cost(
                                            current_Q2, target_Q)
        cost = layers.reduce_mean(cost)
        optimizer = fluid.optimizer.AdamOptimizer(self.critic_lr)
        optimizer.minimize(cost)
        return cost

    def sync_target(self, decay=None):
        if decay is None:
            decay = 1.0 - self.tau
        self.critic.sync_weights_to(self.target_critic, decay=decay)
