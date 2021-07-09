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
import paddle.nn.functional as F
from paddle.distribution import Categorical
from parl.utils.utils import check_model_method
import numpy as np

__all__ = ['A2C']


class A2C(parl.Algorithm):
    def __init__(self, model, vf_loss_coeff=None):
        """ A2C algorithm

        Args:
            model (parl.Model): forward network of policy and value
            vf_loss_coeff (float): coefficient of the value function loss
        """
        # check model and vf_loss_coeff input
        check_model_method(model, 'value', self.__class__.__name__)
        check_model_method(model, 'policy', self.__class__.__name__)
        check_model_method(model, 'policy_and_value', self.__class__.__name__)
        assert isinstance(vf_loss_coeff, (int, float))

        self.model = model
        self.vf_loss_coeff = vf_loss_coeff
        clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=40.0)
        self.optimizer = paddle.optimizer.Adam(
            learning_rate=0.001,
            parameters=self.model.parameters(),
            grad_clip=clip)

    def learn(self, obs, actions, advantages, target_values, learning_rate,
              entropy_coeff):
        """
        Args:
            obs: An float32 tensor of shape ([B] + observation_space).
                 E.g. [B, C, H, W] in atari.
            actions: An int64 tensor of shape [B].
            advantages: A float32 tensor of shape [B].
            target_values: A float32 tensor of shape [B].
            learning_rate: float scalar of leanring rate.
            entropy_coeff: float scalar of entropy coefficient.
        """
        # shape: [B, act_dim]
        logits = self.model.policy(obs)
        act_dim = logits.shape[-1]
        actions_onehot = F.one_hot(actions, act_dim)
        # [B, act_dim] --> [B]
        actions_log_probs = paddle.sum(
            F.log_softmax(logits) * actions_onehot, axis=-1)
        # The policy gradient loss
        pi_loss = -1.0 * paddle.sum(actions_log_probs * advantages)

        # The value function loss
        values = self.model.value(obs)
        delta = values - target_values
        vf_loss = 0.5 * paddle.sum(paddle.square(delta))

        # The entropy loss (We want to maximize entropy, so entropy_ceoff < 0)
        # Using the Categorical just for calculating the entropy.
        # See  https://github.com/PaddlePaddle/Paddle/blob/release/2.0/python/paddle/distribution.py for detail
        policy_distribution = Categorical(logits)
        policy_entropy = policy_distribution.entropy()
        entropy = paddle.sum(policy_entropy)

        total_loss = (
            pi_loss + vf_loss * self.vf_loss_coeff + entropy * entropy_coeff)
        self.optimizer.set_lr(learning_rate)
        total_loss.backward()
        self.optimizer.step()
        self.optimizer.clear_grad()
        return total_loss, pi_loss, vf_loss, entropy

    def prob_and_value(self, obs):
        """
        Args:
            obs: An float32 tensor of shape ([B] + observation_space).
                 E.g. [B, C, H, W] in atari.
        """
        logits, values = self.model.policy_and_value(obs)
        probs = F.softmax(logits)

        return probs, values

    def predict(self, obs):
        """
        Args:
            obs: An float32 tensor of shape ([B] + observation_space).
                 E.g. [B, C, H, W] in atari.
        """
        logits = self.model.policy(obs)
        probs = F.softmax(logits)
        predict_actions = paddle.argmax(probs, 1)
        return predict_actions

    def value(self, obs):
        """
        Args:
            obs: An float32 tensor of shape ([B] + observation_space).
                 E.g. [B, C, H, W] in atari.
        """
        values = self.model.value(obs)
        return values
