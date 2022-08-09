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
                 clip_param,
                 value_loss_coef,
                 entropy_coef,
                 initial_lr,
                 eps=None,
                 max_grad_norm=None,
                 continuous_action=False,
                 clip_vloss=True,
                 use_clipped_value_loss=True):
        """ PPO algorithm

        Args:
            model (parl.Model): forward network of actor and critic.
            clip_param (float): epsilon in clipping loss.
            value_loss_coef (float): value function loss coefficient in the optimization objective.
            entropy_coef (float): policy entropy coefficient in the optimization objective.
            initial_lr (float): learning rate.
            eps (float): Adam optimizer epsilon.
            max_grad_norm (float): max gradient norm for gradient clipping.
            continuous_action (bool): whether or not is continuous action environment.
            clip_vloss (bool): whether or not to use a clipped loss for the value function.
            use_clipped_value_loss (bool): whether or not to use advantages normalization.
        """
        # check model method
        check_model_method(model, 'value', self.__class__.__name__)
        check_model_method(model, 'policy', self.__class__.__name__)

        assert isinstance(clip_param, float)
        assert isinstance(value_loss_coef, float)
        assert isinstance(entropy_coef, float)
        assert isinstance(initial_lr, float)
        self.initial_lr = initial_lr
        self.eps = eps
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.clip_vloss = clip_vloss
        self.continuous_action = continuous_action
        self.use_clipped_value_loss = use_clipped_value_loss

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.initial_lr, eps=self.eps)

    def learn(self,
              batch_obs,
              batch_action,
              batch_value,
              batch_return,
              batch_logprob,
              batch_adv,
              lr=None):
        """ update model with PPO algorithm

        Args:
            batch_obs (torch.Tensor):           shape([batch_size] + obs_shape)
            batch_action (torch.Tensor):        shape([batch_size] + action_shape)
            batch_value (torch.Tensor):         shape([batch_size])
            batch_return (torch.Tensor):        shape([batch_size])
            batch_logprob (torch.Tensor):       shape([batch_size])
            batch_adv (torch.Tensor):           shape([batch_size])
            lr (torch.Tensor):
        Returns:
            v_loss (float): value loss
            pg_loss (float): policy loss
            entropy_loss (float): entropy loss
        """
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
        if self.use_clipped_value_loss:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                mb_advantages.std() + 1e-8)

        # Policy loss
        surr1 = mb_advantages * ratio
        surr2 = mb_advantages * torch.clamp(ratio, 1 - self.clip_param,
                                                1 + self.clip_param)
        pg_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        newvalue = newvalue.view(-1)
        if self.clip_vloss:
            v_loss_unclipped = (newvalue - batch_return)**2
            v_clipped = batch_value + torch.clamp(
                newvalue - batch_value,
                -self.clip_param,
                self.clip_param,
            )
            v_loss_clipped = (v_clipped - batch_return)**2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalue - batch_return)**2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - self.entropy_coef * entropy_loss + v_loss * self.value_loss_coef

        if lr:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return v_loss.item(), pg_loss.item(), entropy_loss.item()

    def sample(self, obs):
        """ Define the sampling process. This function returns the action according to action distribution.
        
        Args:
            obs (torch tensor): observation, shape([batch_size] + obs_shape)
        Returns:
            value (torch tensor): value, shape([batch_size, 1])
            action (torch tensor): action, shape([batch_size] + action_shape)
            action_log_probs (torch tensor): action log probs, shape([batch_size])
            action_entropy (torch tensor): action entropy, shape([batch_size])
        """
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
        """ use the model to predict action

        Args:
            obs (torch tensor): observation, shape([batch_size] + obs_shape)
        Returns:
            action (torch tensor): action, shape([batch_size] + action_shape),
                noted that in the discrete case we take the argmax along the last axis as action
        """
        if self.continuous_action:
            action, _ = self.model.policy(obs)
        else:
            logits = self.model.policy(obs)
            dist = Categorical(logits=logits)
            action = dist.probs.argmax(dim=-1, keepdim=True)
        return action

    def value(self, obs):
        """ use the model to predict obs values

        Args:
            obs (torch tensor): observation, shape([batch_size] + obs_shape)
        Returns:
            value (torch tensor): value of obs, shape([batch_size])
        """
        return self.model.value(obs)
