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
from torch.distributions import Normal, Categorical
from parl.utils.utils import check_model_method

__all__ = ['PPO']


class PPO(parl.Algorithm):
    def __init__(self,
                 model,
                 clip_param=0.1,
                 value_loss_coef=0.5,
                 entropy_coef=0.01,
                 initial_lr=2.5e-4,
                 eps=1e-5,
                 max_grad_norm=0.5,
                 use_clipped_value_loss=True,
                 norm_adv=True,
                 continuous_action=False):
        """ PPO algorithm

        Args:
            model (parl.Model): forward network of actor and critic.
            clip_param (float): epsilon in clipping loss.
            value_loss_coef (float): value function loss coefficient in the optimization objective.
            entropy_coef (float): policy entropy coefficient in the optimization objective.
            initial_lr (float): learning rate.
            eps (float): Adam optimizer epsilon.
            max_grad_norm (float): max gradient norm for gradient clipping.
            use_clipped_value_loss (bool): whether or not to use a clipped loss for the value function.
            norm_adv (bool): whether or not to use advantages normalization.
            continuous_action (bool): whether or not is continuous action environment.
        """
        # check model method
        check_model_method(model, 'value', self.__class__.__name__)
        check_model_method(model, 'policy', self.__class__.__name__)

        assert isinstance(clip_param, float)
        assert isinstance(value_loss_coef, float)
        assert isinstance(entropy_coef, float)
        assert isinstance(initial_lr, float)
        assert isinstance(eps, float)
        assert isinstance(max_grad_norm, float)
        assert isinstance(use_clipped_value_loss, bool)
        assert isinstance(norm_adv, bool)
        assert isinstance(continuous_action, bool)

        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.norm_adv = norm_adv
        self.continuous_action = continuous_action

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=initial_lr, eps=eps)

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
            value_loss (float): value loss
            action_loss (float): policy loss
            entropy_loss (float): entropy loss
        """
        values = self.model.value(batch_obs)
        if self.continuous_action:
            mean, std = self.model.policy(batch_obs)
            dist = Normal(mean, std)
            action_log_probs = dist.log_prob(batch_action).sum(1)
            dist_entropy = dist.entropy().sum(1)
        else:
            logits = self.model.policy(batch_obs)
            dist = Categorical(logits=logits)
            action_log_probs = dist.log_prob(batch_action)
            dist_entropy = dist.entropy()
        entropy_loss = dist_entropy.mean()

        if self.norm_adv:
            batch_adv = (batch_adv - batch_adv.mean()) / (
                batch_adv.std() + 1e-8)

        ratio = torch.exp(action_log_probs - batch_logprob)
        surr1 = ratio * batch_adv
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                            1.0 + self.clip_param) * batch_adv
        action_loss = -torch.min(surr1, surr2).mean()

        values = values.view(-1)
        if self.use_clipped_value_loss:
            value_pred_clipped = batch_value + torch.clamp(
                values - batch_value,
                -self.clip_param,
                self.clip_param,
            )
            value_losses = (values - batch_return).pow(2)
            value_losses_clipped = (value_pred_clipped - batch_return).pow(2)
            value_loss = 0.5 * torch.max(value_losses,
                                         value_losses_clipped).mean()
        else:
            value_loss = 0.5 * (batch_return - values).pow(2).mean()
        loss = value_loss * self.value_loss_coef + action_loss - entropy_loss * self.entropy_coef

        if lr:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return value_loss.item(), action_loss.item(), entropy_loss.item()

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
            mean, std = self.model.policy(obs)
            dist = Normal(mean, std)
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
