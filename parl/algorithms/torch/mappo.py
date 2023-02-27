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

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import parl

__all__ = ['MAPPO', 'PopArt']


class MAPPO(parl.Algorithm):
    def __init__(self,
                 model,
                 clip_param,
                 value_loss_coef,
                 entropy_coef,
                 initial_lr,
                 huber_delta,
                 eps=None,
                 max_grad_norm=None,
                 use_popart=True,
                 use_value_active_masks=True):
        """  MAPPO algorithm

        Args:
            model (parl.Model): model that contains both value network and policy network
            clip_param (float): the clipping strength for value loss clipping
            value_loss_coef (float): the coefficient for value loss
            entropy_coef (float): the coefficient for entropy
            initial_lr (float): initial learning rate.
            huber_delta (float): coefficience of huber loss
            eps (None or float): epsilon for Adam optimizer
            max_grad_norm (float): threshold for grad norm clipping
            use_popart (bool): whether to use PopArt to normalize rewards
            use_value_active_masks (bool): whether to mask useless data in value loss
        """
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.
                                   is_available() else "cpu")
        self.model.to(self.device)
        self.multi_discrete = self.model.actor.multi_discrete
        self.tpdv = dict(dtype=torch.float32, device=self.device)

        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.lr = initial_lr
        self.huber_delta = huber_delta
        self.eps = eps
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self._use_popart = use_popart
        self._use_value_active_masks = use_value_active_masks

        if self._use_popart:
            self.value_normalizer = PopArt(self.model.critic.v_out.weight,
                                           self.model.critic.v_out.bias)
        else:
            self.value_normalizer = None

        self.actor_optimizer = torch.optim.Adam(
            self.model.actor.parameters(), lr=self.lr, eps=self.eps)
        self.critic_optimizer = torch.optim.Adam(
            self.model.critic.parameters(), lr=self.lr, eps=self.eps)

    def sample(self, cent_obs, obs, deterministic=False):
        """ Sample action from parameterized policy
        """
        policy = self.model.policy(obs)

        if not self.multi_discrete:
            action_dis = Categorical(logits=policy)
            if deterministic:
                actions = action_dis.probs.argmax(dim=-1, keepdim=True)
            else:
                actions = action_dis.sample().unsqueeze(-1)
            action_log_probs = action_dis.log_prob(actions.squeeze(-1)).view(
                actions.size(0), -1).sum(-1).unsqueeze(-1)
        else:
            actions = []
            action_log_probs = []
            for i in range(len(self.model.act_dim)):
                action_dis = Categorical(logits=policy[i])
                if deterministic:
                    action = action_dis.probs.argmax(dim=-1, keepdim=True)
                else:
                    action = action_dis.sample().unsqueeze(-1)
                action_log_prob = action_dis.log_prob(action.squeeze(-1)).view(
                    action.size(0), -1).sum(-1).unsqueeze(-1)
                actions.append(action)
                action_log_probs.append(action_log_prob)
            actions = torch.cat(actions, -1)
            action_log_probs = torch.cat(action_log_probs, -1)

        values = self.model.value(cent_obs)
        return values, actions, action_log_probs

    def learn(self,
              share_obs_batch,
              obs_batch,
              actions_batch,
              value_preds_batch,
              return_batch,
              old_action_log_probs_batch,
              adv_targ,
              active_masks_batch=None):
        """ update the value network and policy network parameters.
        """
        policy = self.model.policy(obs_batch)
        if not self.multi_discrete:
            action_dis = Categorical(logits=policy)
            action_log_probs = action_dis.log_prob(
                actions_batch.squeeze(-1)).view(actions_batch.size(0),
                                                -1).sum(-1).unsqueeze(-1)
            if active_masks_batch is not None:
                dist_entropy = (action_dis.entropy() * active_masks_batch.
                                squeeze(-1)).sum() / active_masks_batch.sum()
            else:
                dist_entropy = action_dis.entropy().mean()
        else:
            actions_batch = torch.transpose(actions_batch, 0, 1)
            action_log_probs = []
            dist_entropy = []

            for i in range(len(self.model.act_dim)):
                action_dis = Categorical(logits=policy[i])
                action_log_probs.append(
                    action_dis.log_prob(actions_batch[i].squeeze(-1)).view(
                        actions_batch[i].size(0), -1).sum(-1).unsqueeze(-1))
                if active_masks_batch is not None:
                    dist_entropy.append(
                        (action_dis.entropy() * active_masks_batch.squeeze(-1)
                         ).sum() / active_masks_batch.sum())
                else:
                    dist_entropy.append(action_dis.entropy().mean())
            action_log_probs = torch.cat(action_log_probs, -1)
            dist_entropy = sum(dist_entropy) / len(dist_entropy)

        values = self.model.value(share_obs_batch)

        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param,
                            1.0 + self.clip_param) * adv_targ

        if active_masks_batch is not None:
            policy_action_loss = (
                -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) *
                active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(
                torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss
        self.actor_optimizer.zero_grad()
        (policy_loss - dist_entropy * self.entropy_coef).backward()
        nn.utils.clip_grad_norm_(self.model.actor.parameters(),
                                 self.max_grad_norm)
        self.actor_optimizer.step()

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch,
                                         return_batch, active_masks_batch)
        self.critic_optimizer.zero_grad()
        (value_loss * self.value_loss_coef).backward()
        nn.utils.clip_grad_norm_(self.model.critic.parameters(),
                                 self.max_grad_norm)
        self.critic_optimizer.step()

        return value_loss, policy_loss, dist_entropy

    def cal_value_loss(self,
                       values,
                       value_preds_batch,
                       return_batch,
                       active_masks_batch=None):
        """ Calculate value function loss.
        """
        value_pred_clipped = value_preds_batch + (
            values - value_preds_batch).clamp(-self.clip_param,
                                              self.clip_param)

        if self._use_popart:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(
                return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(
                return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
        value_loss_original = huber_loss(error_original, self.huber_delta)

        value_loss = torch.max(value_loss_original, value_loss_clipped)

        if self._use_value_active_masks:
            value_loss = (value_loss *
                          active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def value(self, cent_obs):
        """ Predict value from parameterized value function
        """
        values = self.model.value(cent_obs)
        return values


class PopArt(torch.nn.Module):
    def __init__(self, weight, bias, norm_axes=1, beta=0.99999, epsilon=1e-5):
        """  PopArt layer use running mean and std to normalize rewards.
             The PopArt code is mainly referenced and copied from https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/algorithms/utils/popart.py

        Args:
            weight (Parameter): output layer weight of critic net
            bias (Parameter): output layer bias of critic net
            norm_axes (float): the number of groups to separate the channels into
            beta (float): the fixed decay rate determining the horizon used to compute the statistics
            epsilon (float): the epsilon value to use to avoid division by zero
        """
        super(PopArt, self).__init__()
        self.beta = beta
        self.epsilon = epsilon
        self.norm_axes = norm_axes
        self.tpdv = dict(
            dtype=torch.float32,
            device=torch.device("cuda:0" if torch.cuda.
                                is_available() else "cpu"))

        self.weight = weight
        self.bias = bias
        output_shape = 1

        self.stddev = nn.Parameter(
            torch.ones(output_shape), requires_grad=False).to(**self.tpdv)
        self.mean = nn.Parameter(
            torch.zeros(output_shape), requires_grad=False).to(**self.tpdv)
        self.mean_sq = nn.Parameter(
            torch.zeros(output_shape), requires_grad=False).to(**self.tpdv)
        self.debiasing_term = nn.Parameter(
            torch.tensor(0.0), requires_grad=False).to(**self.tpdv)

        self.reset_parameters()

    def reset_parameters(self):
        self.mean.zero_()
        self.mean_sq.zero_()
        self.debiasing_term.zero_()

    @torch.no_grad()
    def update(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        old_mean, old_var = self.debiased_mean_var()
        old_stddev = torch.sqrt(old_var)

        batch_mean = input_vector.mean(dim=tuple(range(self.norm_axes)))
        batch_sq_mean = (input_vector
                         **2).mean(dim=tuple(range(self.norm_axes)))

        self.mean.mul_(self.beta).add_(batch_mean * (1.0 - self.beta))
        self.mean_sq.mul_(self.beta).add_(batch_sq_mean * (1.0 - self.beta))
        self.debiasing_term.mul_(self.beta).add_(1.0 * (1.0 - self.beta))

        new_mean, new_var = self.debiased_mean_var()
        new_stddev = torch.sqrt(new_var)

        self.weight.data = self.weight * old_stddev / new_stddev
        self.bias.data = (
            old_stddev * self.bias + old_mean - new_mean) / new_stddev

    def debiased_mean_var(self):
        debiased_mean = self.mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.mean_sq / self.debiasing_term.clamp(
            min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    def normalize(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)
        mean, var = self.debiased_mean_var()
        out = (input_vector - mean[(None, ) * self.norm_axes]
               ) / torch.sqrt(var)[(None, ) * self.norm_axes]
        return out

    def denormalize(self, input_vector):
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)
        mean, var = self.debiased_mean_var()
        out = input_vector * torch.sqrt(var)[(None, ) * self.norm_axes] + mean[
            (None, ) * self.norm_axes]
        out = out.cpu().numpy()
        return out


def huber_loss(e, d):
    """ Huber loss function
    """
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a * e**2 / 2 + b * d * (abs(e) - d / 2)
