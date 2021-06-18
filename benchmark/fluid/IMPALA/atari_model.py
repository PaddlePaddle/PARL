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

import paddle.fluid as fluid
import parl
from parl import layers
from paddle.fluid.param_attr import ParamAttr


class AtariModel(parl.Model):
    def __init__(self, act_dim):

        self.conv1 = layers.conv2d(
            num_filters=16, filter_size=4, stride=2, padding=1, act='relu')
        self.conv2 = layers.conv2d(
            num_filters=32, filter_size=4, stride=2, padding=2, act='relu')
        self.conv3 = layers.conv2d(
            num_filters=256, filter_size=11, stride=1, padding=0, act='relu')

        self.policy_conv = layers.conv2d(
            num_filters=act_dim,
            filter_size=1,
            stride=1,
            padding=0,
            act=None,
            param_attr=ParamAttr(initializer=fluid.initializer.Normal()))

        self.value_fc = layers.fc(
            size=1,
            param_attr=ParamAttr(initializer=fluid.initializer.Normal()))

    def policy(self, obs):
        """
        Args:
            obs: A float32 tensor of shape [B, C, H, W]
        Returns:
            policy_logits: B * ACT_DIM
        """
        obs = obs / 255.0
        conv1 = self.conv1(obs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        policy_conv = self.policy_conv(conv3)
        policy_logits = layers.flatten(policy_conv, axis=1)
        return policy_logits

    def value(self, obs):
        """
        Args:
            obs: A float32 tensor of shape [B, C, H, W]
        Returns:
            value: B
        """
        obs = obs / 255.0
        conv1 = self.conv1(obs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        flatten = layers.flatten(conv3, axis=1)
        value = self.value_fc(flatten)
        value = layers.squeeze(value, axes=[1])
        return value
