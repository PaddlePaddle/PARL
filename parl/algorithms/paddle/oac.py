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
from paddle.distribution import Normal
import paddle.nn.functional as F
from parl.utils.utils import check_model_method
from copy import deepcopy
import math

__all__ = ['OAC']


class OAC(parl.Algorithm):
    def __init__(self,
                 model,
                 gamma=None,
                 tau=None,
                 alpha=None,
                 beta=None,
                 delta=None,
                 actor_lr=None,
                 critic_lr=None):
        """ OAC algorithm

        Args:
            model(parl.Model): forward network of actor and critic.
            gamma(float): discounted factor for reward computation
            tau (float): decay coefficient when updating the weights of self.target_model with self.model
            alpha (float): Temperature parameter determines the relative importance of the entropy against the reward
            beta (float): determines the relative importance of sigma_Q
            delta (float): determines the relative changes of exploration`s mean
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
        assert isinstance(alpha, float)
        assert isinstance(beta, float)
        assert isinstance(delta, float)
        assert isinstance(actor_lr, float)
        assert isinstance(critic_lr, float)

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.model = model
        self.target_model = deepcopy(self.model)
        self.actor_optimizer = paddle.optimizer.Adam(
            learning_rate=actor_lr, parameters=self.model.get_actor_params())
        self.critic_optimizer = paddle.optimizer.Adam(
            learning_rate=critic_lr, parameters=self.model.get_critic_params())

    def predict(self, obs):
        act_mean, _ = self.model.policy(obs)
        action = paddle.tanh(act_mean)
        return action

    def sample(self, obs):
        act_mean, act_log_std = self.model.policy(obs)
        normal = Normal(act_mean, act_log_std.exp())
        # for reparameterization trick  (mean + std*N(0,1))
        x_t = normal.sample([1])
        action = paddle.tanh(x_t)
        log_prob = normal.log_prob(x_t)

        # Enforcing Action Bound
        log_prob -= paddle.log((1 - action.pow(2)) + 1e-6)
        log_prob = paddle.sum(log_prob, axis=-1, keepdim=True)
        return action[0], log_prob[0]

    def get_optimistic_exploration_action(self, obs):
        act_mean, act_log_std = self.model.policy(obs)
        act_std = paddle.exp(act_log_std)
        normal = Normal(act_mean, act_std)
        pre_tanh_mu_T = normal.sample([1])

        tanh_mu_T = paddle.tanh(pre_tanh_mu_T)

        # Get the upper bound of the Q estimate
        Q1, Q2 = self.model.value(obs, tanh_mu_T[0])
        mu_Q = (Q1 + Q2) / 2.0
        sigma_Q = paddle.abs(Q1 - Q2) / 2.0

        Q_UB = mu_Q + self.beta * sigma_Q

        # Obtain the gradient of Q_UB wrt to a with a evaluated at mu_t
        grad = paddle.grad(Q_UB, pre_tanh_mu_T)
        grad = grad[0]
        assert grad is not None
        assert pre_tanh_mu_T.shape == grad.shape

        # Obtain Sigma_T (the covariance of the normal distribution)
        Sigma_T = paddle.pow(act_std, 2)

        # The dividor is (g^T Sigma g) ** 0.5
        # Sigma is diagonal, so this works out to be
        # ( sum_{i=1}^k (g^(i))^2 (sigma^(i))^2 ) ** 0.5
        denom = paddle.sqrt(
            paddle.sum(paddle.multiply(paddle.pow(grad, 2), Sigma_T))) + 10e-6

        # Obtain the change in mu
        mu_C = math.sqrt(2.0 * self.delta) * paddle.multiply(Sigma_T,
                                                             grad) / denom
        assert mu_C.shape == pre_tanh_mu_T.shape

        mu_E = pre_tanh_mu_T + mu_C
        mu_E = paddle.squeeze(mu_E, axis=0)
        # Construct the tanh normal distribution and sample the exploratory action from it
        assert mu_E.shape == act_std.shape

        dist = Normal(mu_E, act_std)
        z = dist.sample([1]).detach()
        action = paddle.tanh(z)
        return action[0]

    def learn(self, obs, action, reward, next_obs, terminal):
        critic_loss = self._critic_learn(obs, action, reward, next_obs,
                                         terminal)
        actor_loss = self._actor_learn(obs)

        self.sync_target()
        return critic_loss, actor_loss

    def _critic_learn(self, obs, action, reward, next_obs, terminal):
        with paddle.no_grad():
            next_action, next_log_pro = self.sample(next_obs)
            q1_next, q2_next = self.target_model.value(next_obs, next_action)
            target_Q = paddle.minimum(q1_next,
                                      q2_next) - self.alpha * next_log_pro
            terminal = paddle.cast(terminal, dtype='float32')
            target_Q = reward + self.gamma * (1. - terminal) * target_Q
        cur_q1, cur_q2 = self.model.value(obs, action)

        critic_loss = F.mse_loss(cur_q1, target_Q) + F.mse_loss(
            cur_q2, target_Q)

        self.critic_optimizer.clear_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss

    def _actor_learn(self, obs):
        act, log_pi = self.sample(obs)
        q1_pi, q2_pi = self.model.value(obs, act)
        min_q_pi = paddle.minimum(q1_pi, q2_pi)
        actor_loss = ((self.alpha * log_pi) - min_q_pi).mean()

        self.actor_optimizer.clear_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss

    def sync_target(self, decay=None):
        if decay is None:
            decay = 1.0 - self.tau
        self.model.sync_weights_to(self.target_model, decay=decay)
