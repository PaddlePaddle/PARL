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

import torch.nn as nn
import parl


class AtariModel(parl.Model):
    """ Neural Network to solve Atari problem.

    Args:
        act_dim (int): Dimension of action space.
        dueling (bool): True if use dueling architecture else False
    """

    def __init__(self, act_dim, dueling=False):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=4, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=2)
        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=1,
            padding=1)
        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        self.dueling = dueling

        if dueling:
            self.linear_1_adv = nn.Linear(in_features=6400, out_features=512)
            self.linear_2_adv = nn.Linear(
                in_features=512, out_features=act_dim)
            self.linear_1_val = nn.Linear(in_features=6400, out_features=512)
            self.linear_2_val = nn.Linear(in_features=512, out_features=1)

        else:
            self.linear_1 = nn.Linear(in_features=6400, out_features=act_dim)

        self.reset_params()

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, obs):
        """ Perform forward pass 

        Args:
            obs (torch.Tensor): shape of (batch_size, 3, 84, 84), mini batch of observations
        """
        obs = obs / 255.0
        out = self.max_pool(self.relu(self.conv1(obs)))
        out = self.max_pool(self.relu(self.conv2(out)))
        out = self.max_pool(self.relu(self.conv3(out)))
        out = self.relu(self.conv4(out))
        out = self.flatten(out)

        if self.dueling:
            As = self.relu(self.linear_1_adv(out))
            As = self.linear_2_adv(As)
            V = self.relu(self.linear_1_val(out))
            V = self.linear_2_val(V)
            Q = As + (V - As.mean(dim=1, keepdim=True))

        else:
            Q = self.linear_1(out)

        return Q
