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
import paddle.optimizer as optim
import paddle.nn as nn
from paddle.distribution import Normal
from parl.utils.utils import check_model_method

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
        """ PPO algorithm

        Args:
            model (parl.Model): model that contains both value network and policy network
            clip_param (float): the clipping strength for value loss clipping
            value_loss_coef (float): the coefficient for value loss (c_1)
            entropy_coef (float): the coefficient for entropy (c_2)
            initial_lr (float): initial learning rate.
            eps (None or float): epsilon for Adam optimizer
            max_grad_norm (float): threshold for grad norm clipping
            use_clipped_value_loss (bool): whether use value loss clipping
        """
        # checks
        check_model_method(model, 'value', self.__class__.__name__)
        check_model_method(model, 'policy', self.__class__.__name__)

        super().__init__(model)
        self.clip_param = clip_param

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.use_clipped_value_loss = use_clipped_value_loss
        clip = nn.ClipGradByNorm(max_grad_norm)

        self.optimizer = optim.Adam(
            parameters=model.parameters(),
            learning_rate=initial_lr,
            epsilon=eps,
            grad_clip=clip)

    def learn(self, obs_batch, actions_batch, value_preds_batch, return_batch,
              old_action_log_probs_batch, adv_targ):
        """ update the value network and policy network parameters.
        """
        values = self.model.value(obs_batch)

        # log std so the std is always positive after e^{log_std}
        mean, log_std = self.model.policy(obs_batch)
        dist = Normal(mean, log_std.exp())

        # Continuous actions are usually considered to be independent,
        # so we can sum components of the ``log_prob`` or the entropy.
        action_log_probs = dist.log_prob(actions_batch).sum(
            axis=-1, keepdim=True)
        dist_entropy = dist.entropy().sum(axis=-1).mean()

        ratio = paddle.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = ratio * adv_targ
        surr2 = paddle.clip(ratio, 1.0 - self.clip_param,
                            1.0 + self.clip_param) * adv_targ
        action_loss = -paddle.minimum(surr1, surr2).mean()

        # calculate value loss using semi gradient TD
        if self.use_clipped_value_loss:
            value_pred_clipped = value_preds_batch + \
                (values - value_preds_batch).clip(-self.clip_param, self.clip_param)
            value_losses = (values - return_batch).pow(2)
            value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
            value_loss = 0.5 * paddle.maximum(value_losses,
                                              value_losses_clipped).mean()
        else:
            value_loss = 0.5 * (return_batch - values).pow(2).mean()

        (value_loss * self.value_loss_coef + action_loss -
         dist_entropy * self.entropy_coef).backward()
        self.optimizer.step()
        self.optimizer.clear_grad()

        return value_loss.numpy(), action_loss.numpy(), dist_entropy.numpy()

    def sample(self, obs):
        """ Sample action from parameterized policy
        """
        value = self.model.value(obs)
        mean, log_std = self.model.policy(obs)
        dist = Normal(mean, log_std.exp())
        action = dist.sample([1])
        action_log_probs = dist.log_prob(action).sum(-1, keepdim=True)

        return value, action, action_log_probs

    def predict(self, obs):
        """ Predict action from parameterized policy, action with maximum probability is selected as greedy action
        """
        mean, _ = self.model.policy(obs)
        return mean

    def value(self, obs):
        """ Predict value from parameterized value function
        """
        return self.model.value(obs)
