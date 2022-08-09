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
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.optimizer as optim
from parl.utils.utils import check_model_method
from paddle.distribution import Normal, Categorical

__all__ = ['PPO']


class PPO(parl.Algorithm):
    def __init__(self,
                 model,
                 clip_param=None,
                 value_loss_coef=None,
                 entropy_coef=None,
                 initial_lr=None,
                 eps=None,
                 max_grad_norm=None,
                 continuous_action=False,
                 clip_vloss=True,
                 norm_adv=True):
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
            norm_adv (bool): whether or not to use advantages normalization.
        """
        # check model methods
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
        self.norm_adv = norm_adv

        self.model = model

        clip = nn.ClipGradByNorm(self.max_grad_norm)
        self.optimizer = optim.Adam(
            parameters=self.model.parameters(),
            learning_rate=self.initial_lr,
            epsilon=self.eps,
            grad_clip=clip)

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

            act_dim = logits.shape[-1]
            batch_action = paddle.to_tensor(batch_action, dtype='int64')
            actions_onehot = F.one_hot(batch_action, act_dim)

            newlogprob = paddle.sum(
                F.log_softmax(logits) * actions_onehot, axis=-1)
            entropy = dist.entropy()

        logratio = newlogprob - batch_logprob
        ratio = logratio.exp()

        mb_advantages = batch_adv
        if self.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                mb_advantages.std() + 1e-8)

        # Policy loss
        pg_loss1 = mb_advantages * ratio
        pg_loss2 = mb_advantages * paddle.clip(ratio, 1 - self.clip_param,
                                               1 + self.clip_param)
        pg_loss = -paddle.minimum(pg_loss1, pg_loss2).mean()

        # Value loss
        newvalue = newvalue.reshape([-1])
        if self.clip_vloss:
            v_loss_unclipped = (newvalue - batch_return)**2
            v_clipped = batch_value + paddle.clip(
                newvalue - batch_value,
                -self.clip_param,
                self.clip_param,
            )
            v_loss_clipped = (v_clipped - batch_return)**2
            v_loss_max = paddle.maximum(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalue - batch_return)**2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - self.entropy_coef * entropy_loss + v_loss * self.value_loss_coef

        if lr:
            self.optimizer.set_lr(lr)

        loss.backward()
        self.optimizer.step()
        self.optimizer.clear_grad()

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
            action = dist.sample([1])

            action_log_probs = dist.log_prob(action).sum(-1)
            action_entropy = dist.entropy().sum(-1).mean()
        else:
            logits = self.model.policy(obs)
            dist = Categorical(logits=logits)
            action = dist.sample([1])

            act_dim = logits.shape[-1]
            actions_onehot = F.one_hot(action, act_dim)
            action_log_probs = paddle.sum(
                F.log_softmax(logits) * actions_onehot, axis=-1)
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
            probs = F.softmax(logits)
            action = paddle.argmax(probs, 1)
        return action

    def value(self, obs):
        """ use the model to predict obs values

        Args:
            obs (torch tensor): observation, shape([batch_size] + obs_shape)
        Returns:
            value (torch tensor): value of obs, shape([batch_size])
        """
        return self.model.value(obs)
