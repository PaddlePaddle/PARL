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
import parl
import paddle.nn as nn
import paddle.nn.functional as F


class AtariModel(parl.Model):
    def __init__(self, act_dim):
        super(AtariModel, self).__init__()
        self.conv1 = nn.Conv2D(
            in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=1)
        self.conv2 = nn.Conv2D(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=2)
        self.conv3 = nn.Conv2D(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=0)

        self.flatten = nn.Flatten()

        # Need to calc the size of the in_features according to the input image.
        # The default size of the input image is 84 * 84
        self.fc = nn.Linear(in_features=64 * 9 * 9, out_features=512)

        self.policy_fc = nn.Linear(in_features=512, out_features=act_dim)
        self.value_fc = nn.Linear(in_features=512, out_features=1)

    def policy(self, obs):
        """
        Args:
            obs: A float32 tensor array of shape [B, C, H, W]

        Returns:
            policy_logits: B * ACT_DIM
        """
        obs = obs / 255.0
        conv1 = F.relu(self.conv1(obs))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        flatten = self.flatten(conv3)
        fc_output = F.relu(self.fc(flatten))
        policy_logits = self.policy_fc(fc_output)

        return policy_logits

    def value(self, obs):
        """
        Args:
            obs: A float32 tensor of shape [B, C, H, W]

        Returns:
            values: B
        """
        obs = obs / 255.0
        conv1 = F.relu(self.conv1(obs))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        flatten = self.flatten(conv3)
        fc_output = F.relu(self.fc(flatten))
        values = self.value_fc(fc_output)
        values = paddle.squeeze(values, axis=1)
        return values

    def policy_and_value(self, obs):
        """
        Args:
            obs: A tensor array of shape [B, C, H, W]

        Returns:
            policy_logits: B * ACT_DIM
            values: B
        """
        obs = obs / 255.0
        conv1 = F.relu(self.conv1(obs))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        flatten = self.flatten(conv3)
        fc_output = F.relu(self.fc(flatten))

        policy_logits = self.policy_fc(fc_output)

        values = self.value_fc(fc_output)
        values = paddle.squeeze(values, axis=1)
        return policy_logits, values
