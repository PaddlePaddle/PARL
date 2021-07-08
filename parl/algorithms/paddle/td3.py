#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import parl
import paddle
import paddle.nn.functional as F
from parl.utils.utils import check_model_method
from copy import deepcopy

__all__ = ['TD3']


class TD3(parl.Algorithm):
    def __init__(self,
                 model,
                 gamma=None,
                 tau=None,
                 actor_lr=None,
                 critic_lr=None,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 policy_freq=2):
        """ TD3 algorithm

        Args:
            model(parl.Model): forward network of actor and critic.
            gamma(float): discounted factor for reward computation
            tau (float): decay coefficient when updating the weights of self.target_model with self.model
            actor_lr (float): learning rate of the actor model
            critic_lr (float): learning rate of the critic model
            policy_noise(float): noise added to target policy during critic update
            noise_clip(float): range to clip target policy noise
            policy_freq(int): frequency of delayed policy updates
        """
        # checks
        check_model_method(model, 'value', self.__class__.__name__)
        check_model_method(model, 'policy', self.__class__.__name__)
        check_model_method(model, 'Q1', self.__class__.__name__)
        check_model_method(model, 'get_actor_params', self.__class__.__name__)
        check_model_method(model, 'get_critic_params', self.__class__.__name__)
        assert isinstance(gamma, float)
        assert isinstance(tau, float)
        assert isinstance(actor_lr, float)
        assert isinstance(critic_lr, float)

        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0

        self.model = model
        self.target_model = deepcopy(self.model)
        self.actor_optimizer = paddle.optimizer.Adam(
            learning_rate=actor_lr, parameters=self.model.get_actor_params())
        self.critic_optimizer = paddle.optimizer.Adam(
            learning_rate=critic_lr, parameters=self.model.get_critic_params())

    def predict(self, obs):
        action = self.model.policy(obs)
        return action

    def learn(self, obs, action, reward, next_obs, terminal):
        self.total_it += 1
        critic_loss = self._critic_learn(obs, action, reward, next_obs,
                                         terminal)
        if self.total_it % self.policy_freq == 0:
            actor_loss = self._actor_learn(obs)
        return critic_loss

    def _critic_learn(self, obs, action, reward, next_obs, terminal):
        with paddle.no_grad():
            noise = paddle.randn(action.shape) * self.policy_noise
            noise = paddle.clip(noise, -self.noise_clip, self.noise_clip)

            next_action = self.target_model.policy(next_obs) + noise
            next_action = paddle.clip(next_action, -1., 1.)

            target_q1, target_q2 = self.target_model.value(
                next_obs, next_action)
            target_q = paddle.minimum(target_q1, target_q2)
            terminal = paddle.cast(terminal, dtype='float32')
            target_q = reward + (1. - terminal) * self.gamma * target_q
        current_q1, current_q2 = self.model.value(obs, action)

        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(
            current_q2, target_q)

        self.critic_optimizer.clear_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss

    def _actor_learn(self, obs):
        actor_loss = -self.model.Q1(obs, self.model.policy(obs)).mean()

        self.actor_optimizer.clear_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.sync_target()
        return actor_loss

    def sync_target(self, decay=None):
        if decay is None:
            decay = 1. - self.tau
        self.model.sync_weights_to(self.target_model, decay=decay)
