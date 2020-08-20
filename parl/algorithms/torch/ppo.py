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

import parl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

__all__ = ['PPO']


class PPO(parl.Algorithm):
    def __init__(self,
                 model,
                 clip_param,
                 value_loss_coef,
                 entropy_coef,
                 initial_lr,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True):
        self.model = model

        self.clip_param = clip_param

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(model.parameters(), lr=initial_lr, eps=eps)

    def learn(self, obs_batch, actions_batch, value_preds_batch, return_batch,
              old_action_log_probs_batch, adv_targ):
        values = self.model.value(obs_batch)
        mean, log_std = self.model.policy(obs_batch)
        dist = Normal(mean, log_std.exp())

        action_log_probs = dist.log_prob(actions_batch).sum(-1, keepdim=True)
        dist_entropy = dist.entropy().sum(-1).mean()

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

        self.optimizer.zero_grad()
        (value_loss * self.value_loss_coef + action_loss -
         dist_entropy * self.entropy_coef).backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item()

    def sample(self, obs):
        value = self.model.value(obs)
        mean, log_std = self.model.policy(obs)
        dist = Normal(mean, log_std.exp())
        action = dist.sample()
        action_log_probs = dist.log_prob(action).sum(-1, keepdim=True)

        return value, action, action_log_probs

    def predict(self, obs):
        mean, _ = self.model.policy(obs)
        return mean

    def value(self, obs):
        return self.model.value(obs)
