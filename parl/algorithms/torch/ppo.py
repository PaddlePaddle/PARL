#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from parl.utils.utils import check_model_method
from torch.distributions import Normal, Categorical

__all__ = ['PPO']


class PPO(parl.Algorithm):
    def __init__(self,
                 model,
                 clip_coef=None,
                 vf_coef=None,
                 ent_coef=None,
                 start_lr=None,
                 eps=None,
                 max_grad_norm=None,
                 clip_vloss=True,
                 norm_adv=True):
        """ PPO algorithm
            Args:
                model(parl.Model): forward network of actor and critic.
        """
        # checks
        check_model_method(model, 'value', self.__class__.__name__)
        check_model_method(model, 'policy', self.__class__.__name__)

        assert isinstance(clip_coef, float)
        assert isinstance(vf_coef, float)
        assert isinstance(ent_coef, float)
        assert isinstance(start_lr, float)
        self.start_lr = start_lr
        self.eps = eps
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.clip_vloss = clip_vloss

        self.norm_adv = norm_adv

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device)
        self.continuous_action = self.model.continuous_action
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.start_lr, eps=self.eps)

    def learn(self,
              batch_obs,
              batch_action,
              batch_value,
              batch_return,
              batch_logprob,
              batch_adv,
              lr=None):
        newvalue = self.model.value(batch_obs)
        if self.continuous_action:
            action_mean, action_std = self.model.policy(batch_obs)
            dist = Normal(action_mean, action_std)
            newlogprob = dist.log_prob(batch_action).sum(1)
            entropy = dist.entropy().sum(1)
        else:
            logits = self.model.policy(batch_obs)
            dist = Categorical(logits=logits)
            newlogprob = dist.log_prob(batch_action)
            entropy = dist.entropy()

        logratio = newlogprob - batch_logprob
        ratio = logratio.exp()

        mb_advantages = batch_adv
        if self.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                mb_advantages.std() + 1e-8)

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef,
                                                1 + self.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        newvalue = newvalue.view(-1)
        if self.clip_vloss:
            v_loss_unclipped = (newvalue - batch_return)**2
            v_clipped = batch_value + torch.clamp(
                newvalue - batch_value,
                -self.clip_coef,
                self.clip_coef,
            )
            v_loss_clipped = (v_clipped - batch_return)**2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalue - batch_return)**2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

        if lr:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return v_loss.item(), pg_loss.item(), entropy_loss.item()

    def sample(self, obs):
        value = self.model.value(obs)

        if self.continuous_action:
            action_mean, action_std = self.model.policy(obs)
            dist = Normal(action_mean, action_std)
            action = dist.sample()

            action_log_probs = dist.log_prob(action).sum(1)
            action_entropy = dist.entropy().sum(1)
        else:
            logits = self.model.policy(obs)
            dist = Categorical(logits=logits)
            action = dist.sample()

            action_log_probs = dist.log_prob(action)
            action_entropy = dist.entropy()

        return value, action, action_log_probs, action_entropy

    def predict(self, obs):
        if self.continuous_action:
            action, _ = self.model.policy(obs)
        else:
            logits = self.model.policy(obs)
            dist = Categorical(logits=logits)
            action = dist.probs.argmax(dim=-1, keepdim=True)
        return action

    def value(self, obs):
        return self.model.value(obs)
