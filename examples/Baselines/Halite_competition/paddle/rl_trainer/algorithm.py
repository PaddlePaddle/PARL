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

import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from paddle.distribution import Categorical


class PPO:
    def __init__(self,
                 actor,
                 critic,
                 clip_param,
                 value_loss_coef,
                 entropy_coef,
                 initial_lr,
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

        self.optim = optim.Adam(
            parameters=list(self.actor.parameters()) + list(
                self.critic.parameters()),
            learning_rate=initial_lr,
            grad_clip=nn.ClipGradByNorm(max_grad_norm))

    def learn(self, obs_batch, actions_batch, value_preds_batch, return_batch,
              old_action_log_probs_batch, adv_targ):
        """Update rule for ppo algorithm
        Args:
            obs_batch (paddle.tensor): a batch of states
            actions_batch (paddle.tensor): a batch of actions
            value_preds_batch (paddle.tensor): a batch of predicted state value
            return_batch (paddle.tensor): a batch of discounted return
            old_action_log_probs_batch (paddle.tensor): a batch of log prob of old actions
            adv_targ (paddle.tensor): a batch of advantage value
        """

        values = self.critic(obs_batch)
        probs = self.actor(obs_batch)

        dist = Categorical(probs)
        action_log_probs = dist.log_prob(actions_batch)
        dist_entropy = dist.entropy().mean()

        ratio = paddle.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = ratio * adv_targ
        surr2 = paddle.clip(ratio, 1.0 - self.clip_param,
                            1.0 + self.clip_param) * adv_targ
        action_loss = -paddle.minimum(surr1, surr2).mean()

        if self.use_clipped_value_loss:
            value_pred_clipped = value_preds_batch + \
                (values - value_preds_batch).clip(-self.clip_param, self.clip_param)
            value_losses = (values - return_batch).pow(2)
            value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
            value_loss = 0.5 * paddle.maximum(value_losses,
                                              value_losses_clipped).mean()
        else:
            value_loss = 0.5 * (return_batch - values).pow(2).mean()

        self.optim.clear_grad()
        (value_loss + action_loss -
         dist_entropy * self.entropy_coef).backward()
        self.optim.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item()

    def sample(self, obs):
        """Sample action
        Args:
            obs (paddle.tensor): observation
        Return:
            value (paddle.tensor): predicted state value
            action (paddle.tensor): actions sampled from a distribution
            action_log_probs (paddle.tensor): the log probabilites of action
        """

        batch_size = obs.shape[0]
        with paddle.no_grad():
            value = self.critic(obs).squeeze(1)
            action_probs = self.actor(obs)
            dist = Categorical(action_probs)
            action = dist.sample([1]).reshape((batch_size, 1))
            action_log_probs = dist.log_prob(action)

        action = action.squeeze(1)
        action_log_probs = action_log_probs.squeeze(1)

        return value, action, action_log_probs

    def predict(self, obs):
        """Predict action
        Args:
            obs (paddle.tensor): observation
        Return:
            action (paddle.tensor): actions of the highest probability
        """
        with paddle.no_grad():
            action_probs = self.actor(obs)
        return action_probs.argmax(1)

    def value(self, obs):
        """Predict state value
        Args:
            obs (paddle.tensor): observation
        Return:
            value (paddle.tensor): the predicted state value
        """
        with paddle.no_grad():
            value = self.critic(obs)
        value = value.squeeze(1)
        return value
