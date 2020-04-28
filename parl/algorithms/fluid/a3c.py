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

import warnings
warnings.simplefilter('default')

import paddle.fluid as fluid
from parl.core.fluid import layers
from parl.core.fluid.algorithm import Algorithm
from parl.core.fluid.policy_distribution import CategoricalDistribution

__all__ = ['A3C']


class A3C(Algorithm):
    def __init__(self, model, vf_loss_coeff=None):
        """ A3C/A2C algorithm
        
        Args:
            model (parl.Model): forward network of policy and value
            vf_loss_coeff (float): coefficient of the value function loss
        """

        self.model = model
        assert isinstance(vf_loss_coeff, (int, float))
        self.vf_loss_coeff = vf_loss_coeff

    def learn(self, obs, actions, advantages, target_values, learning_rate,
              entropy_coeff):
        """
        Args:
            obs: An float32 tensor of shape ([B] + observation_space).
                 E.g. [B, C, H, W] in atari.
            actions: An int64 tensor of shape [B].
            advantages: A float32 tensor of shape [B].
            target_values: A float32 tensor of shape [B].
            learning_rate: float scalar of learning rate.
            entropy_coeff: float scalar of entropy coefficient.
        """
        logits = self.model.policy(obs)
        policy_distribution = CategoricalDistribution(logits)
        actions_log_probs = policy_distribution.logp(actions)

        # The policy gradient loss
        pi_loss = -1.0 * layers.reduce_sum(actions_log_probs * advantages)

        # The value function loss
        values = self.model.value(obs)
        delta = values - target_values
        vf_loss = 0.5 * layers.reduce_sum(layers.square(delta))

        # The entropy loss (We want to maximize entropy, so entropy_ceoff < 0)
        policy_entropy = policy_distribution.entropy()
        entropy = layers.reduce_sum(policy_entropy)

        total_loss = (
            pi_loss + vf_loss * self.vf_loss_coeff + entropy * entropy_coeff)

        fluid.clip.set_gradient_clip(
            clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=40.0))

        optimizer = fluid.optimizer.AdamOptimizer(learning_rate)
        optimizer.minimize(total_loss)

        return total_loss, pi_loss, vf_loss, entropy

    def sample(self, obs):
        """
        Args:
            obs: An float32 tensor of shape ([B] + observation_space).
                 E.g. [B, C, H, W] in atari.
        """
        logits, values = self.model.policy_and_value(obs)

        policy_dist = CategoricalDistribution(logits)
        sample_actions = policy_dist.sample()

        return sample_actions, values

    def predict(self, obs):
        """
        Args:
            obs: An float32 tensor of shape ([B] + observation_space).
                 E.g. [B, C, H, W] in atari.
        """
        logits = self.model.policy(obs)
        probs = layers.softmax(logits)

        predict_actions = layers.argmax(probs, 1)

        return predict_actions

    def value(self, obs):
        """
        Args:
            obs: An float32 tensor of shape ([B] + observation_space).
                 E.g. [B, C, H, W] in atari.
        """
        values = self.model.value(obs)
        return values
