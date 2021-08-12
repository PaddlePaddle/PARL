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

import torch
import torch.nn as nn
import torch.optim as optim

import parl
from torch.distributions import Categorical


class PPO(parl.Algorithm):
    def __init__(self,
                 actor,
                 critic,
                 clip_param,
                 value_loss_coef,
                 entropy_coef,
                 initial_lr,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=False):
        """PPO algorithm for discrete actions by paddlepaddle
        Args:
            actor (parl.Model): actor network
            critic (parl.Model): critic network
            clip_param (float): param for clipping the importance sampling ratio
            value_loss_coef (float): coefficient for value loss
            entropy_coef (float): coefficient for entropy
            initial_lr (float): learning rate
            max_grad_norm (int): to clip the gradient of network
            use_clipped_value_loss: whether to clip the value loss
        """
        self.actor = actor
        self.critic = critic
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=initial_lr)
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=initial_lr)

    def learn(self, obs_batch, actions_batch, value_preds_batch, return_batch,
              old_action_log_probs_batch, adv_targ):
        """Update rule for ppo algorithm
        Args:
            obs_batch (torhc.tensor): a batch of states
            actions_batch (torch.tensor): a batch of actions
            value_preds_batch (torch.tensor): a batch of predicted state value
            return_batch (torch.tensor): a batch of discounted return
            old_action_log_probs_batch (torch.tensor): a batch of log prob of old actions
            adv_targ (torch.tensor): a batch of advantage value
        """
        values = self.critic(obs_batch)
        probs = self.actor(obs_batch)
        dist = Categorical(probs)
        action_log_probs = dist.log_prob(actions_batch)
        dist_entropy = dist.entropy().mean()

        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                            1.0 + self.clip_param) * adv_targ
        action_loss = -torch.min(surr1, surr2).mean()

        if self.use_clipped_value_loss:
            value_pred_clipped = value_preds_batch + \
                (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
            value_losses = (values - return_batch).pow(2)
            value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
            value_loss = 0.5 * torch.max(value_losses,
                                         value_losses_clipped).mean()
        else:
            value_loss = 0.5 * (return_batch - values).pow(2).mean()

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        #nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        (action_loss - dist_entropy * self.entropy_coef).backward()
        #nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item()

    def sample(self, obs):
        """Sample action
        Args:
            obs (torch.tensor): observation
        Return:
            value (torch.tensor): predicted state value
            action (torch.tensor): actions sampled from a distribution
            action_log_probs (torch.tensor): the log probabilites of action
        """
        with torch.no_grad():
            value = self.critic(obs)
            action_probs = self.actor(obs)
            dist = Categorical(action_probs)
            action = dist.sample()
            action_log_probs = dist.log_prob(action)
        return value, action, action_log_probs

    def predict(self, obs):
        """Predict action
        Args:
            obs (torch.tensor): observation
        Return:
            action (torch.tensor): actions of the highest probability
        """
        with torch.no_grad():
            action_probs = self.actor(obs)
        return action_probs.argmax(1)

    def value(self, obs):
        """Predict state value
        Args:
            obs (torch.tensor): observation
        Return:
            value (torch.tensor): the predicted state value
        """
        with torch.no_grad():
            value = self.critic(obs)
        return value
