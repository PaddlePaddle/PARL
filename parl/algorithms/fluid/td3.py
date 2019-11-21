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
from paddle import fluid
from parl.core.fluid.algorithm import Algorithm

__all__ = ['TD3']


class TD3(Algorithm):
    def __init__(
            self,
            model,
            max_action,
            gamma=None,
            tau=None,
            actor_lr=None,
            critic_lr=None,
            policy_noise=0.2,  # Noise added to target policy during critic update
            noise_clip=0.5,  # Range to clip target policy noise
            policy_freq=2):  # Frequency of delayed policy updates
        assert isinstance(gamma, float)
        assert isinstance(tau, float)
        assert isinstance(actor_lr, float)
        assert isinstance(critic_lr, float)
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.model = model
        self.target_model = deepcopy(model)

    def predict(self, obs):
        """ use actor model of self.model to predict the action
        """
        return self.model.policy(obs)

    def actor_learn(self, obs):
        action = self.model.policy(obs)
        Q = self.model.Q1(obs, action)
        cost = layers.reduce_mean(-1.0 * Q)
        optimizer = fluid.optimizer.AdamOptimizer(self.actor_lr)
        optimizer.minimize(cost, parameter_list=self.model.get_actor_params())
        return cost

    def critic_learn(self, obs, action, reward, next_obs, terminal):
        noise = layers.gaussian_random_batch_size_like(
            action, shape=[-1, action.shape[1]])
        noise = layers.clip(
            noise * self.policy_noise,
            min=-self.noise_clip,
            max=self.noise_clip)
        next_action = self.target_model.policy(next_obs) + noise
        next_action = layers.clip(next_action, -self.max_action,
                                  self.max_action)

        next_Q1, next_Q2 = self.target_model.value(next_obs, next_action)
        next_Q = layers.elementwise_min(next_Q1, next_Q2)

        terminal = layers.cast(terminal, dtype='float32')
        target_Q = reward + (1.0 - terminal) * self.gamma * next_Q
        target_Q.stop_gradient = True

        current_Q1, current_Q2 = self.model.value(obs, action)
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
        self.model.sync_weights_to(self.target_model, decay=decay)
