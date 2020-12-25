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

import parl
import torch
from torch.distributions import Normal
import torch.nn.functional as F
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

__all__ = ['SAC']


class SAC(parl.Algorithm):
    def __init__(self,
                 model,
                 max_action,
                 discount=None,
                 tau=None,
                 alpha=None,
                 actor_lr=None,
                 critic_lr=None,
                 decay=None,
                 policy_freq=1,
                 automatic_entropy_tuning=False,
                 entropy_lr=None,
                 action_space=None):
        assert isinstance(discount, float)
        assert isinstance(tau, float)
        assert isinstance(alpha, float)
        assert isinstance(actor_lr, float)
        assert isinstance(critic_lr, float)
        assert isinstance(entropy_lr, float)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.alpha = alpha
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.decay = decay
        self.policy_freq = policy_freq
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.entropy_lr = entropy_lr
        self.total_it = 0

        self.model = model.to(device)
        self.target_model = deepcopy(self.model)
        self.actor_optimizer = torch.optim.Adam(
            self.model.get_actor_params(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            self.model.get_critic_params(), lr=critic_lr)

        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(
                torch.Tensor(action_space).to(device))
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=self.entropy_lr)

    def predict(self, obs):
        act_mean, act_log_std = self.model.policy(obs)
        normal = Normal(act_mean, act_log_std.exp())
        # for reparameterization trick  (mean + std*N(0,1))
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action

        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdims=True)
        return action, log_prob

    def learn(self, obs, action, reward, next_obs, terminal):
        self._critic_learn(obs, action, reward, next_obs, terminal)
        self._actor_learn(obs)

    def _critic_learn(self, obs, action, reward, next_obs, terminal):
        self.total_it += 1

        with torch.no_grad():
            next_action, next_log_pro = self.predict(next_obs)
            q1_next, q2_next = self.target_model.critic_model(
                next_obs, next_action)
            target_Q = torch.min(q1_next, q2_next) - self.alpha * next_log_pro
            target_Q = reward + self.discount * terminal * target_Q
        cur_q1, cur_q2 = self.model.critic_model(obs, action)

        critic_loss = F.mse_loss(cur_q1, target_Q) + F.mse_loss(
            cur_q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def _actor_learn(self, obs):
        if self.total_it % self.policy_freq == 0:
            act, log_pi = self.predict(obs)
            q1_pi, q2_pi = self.model.critic_model(obs, act)
            min_q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = ((self.alpha * log_pi) - min_q_pi).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.sync_target(decay=self.decay)

            if self.automatic_entropy_tuning is True:
                alpha_loss = -(self.log_alpha *
                               (log_pi + self.target_entropy).detach()).mean()

                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                self.alpha = self.log_alpha.exp()

    def sync_target(self, decay=None):
        if decay is None:
            decay = 1.0 - self.tau
        for param, target_param in zip(self.model.parameters(),
                                       self.target_model.parameters()):
            target_param.data.copy_((1 - decay) * param.data +
                                    decay * target_param.data)
