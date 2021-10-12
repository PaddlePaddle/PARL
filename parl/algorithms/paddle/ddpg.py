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
from copy import deepcopy
from parl.utils.utils import check_model_method

__all__ = ['DDPG']


class DDPG(parl.Algorithm):
    def __init__(self,
                 model,
                 gamma=None,
                 tau=None,
                 actor_lr=None,
                 critic_lr=None):
        """ DDPG algorithm

        Args:
            model(parl.Model): forward network of actor and critic.
            gamma(float): discounted factor for reward computation
            tau (float): decay coefficient when updating the weights of self.target_model with self.model
            actor_lr (float): learning rate of the actor model
            critic_lr (float): learning rate of the critic model
        """
        # checks
        check_model_method(model, 'value', self.__class__.__name__)
        check_model_method(model, 'policy', self.__class__.__name__)
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

        self.model = model
        self.target_model = deepcopy(self.model)
        self.actor_optimizer = paddle.optimizer.Adam(
            learning_rate=actor_lr, parameters=self.model.get_actor_params())
        self.critic_optimizer = paddle.optimizer.Adam(
            learning_rate=critic_lr, parameters=self.model.get_critic_params())

    def predict(self, obs):
        return self.model.policy(obs)

    def learn(self, obs, action, reward, next_obs, terminal):
        critic_loss = self._critic_learn(obs, action, reward, next_obs,
                                         terminal)
        actor_loss = self._actor_learn(obs)

        self.sync_target()
        return critic_loss, actor_loss

    def _critic_learn(self, obs, action, reward, next_obs, terminal):
        with paddle.no_grad():
            # Compute the target Q value
            target_Q = self.target_model.value(
                next_obs, self.target_model.policy(next_obs))
            terminal = paddle.cast(terminal, dtype='float32')
            target_Q = reward + ((1. - terminal) * self.gamma * target_Q)

        # Get current Q estimate
        current_Q = self.model.value(obs, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.clear_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss

    def _actor_learn(self, obs):
        # Compute actor loss and Update the frozen target models
        actor_loss = -self.model.value(obs, self.model.policy(obs)).mean()

        # Optimize the actor
        self.actor_optimizer.clear_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss

    def sync_target(self, decay=None):
        """ update the target network with the training network

        Args:
            decay(float): the decaying factor while updating the target network with the training network.
                        0 represents the **assignment**. None represents updating the target network slowly that depends on the hyperparameter `tau`.
        """
        if decay is None:
            decay = 1.0 - self.tau
        self.model.sync_weights_to(self.target_model, decay=decay)
