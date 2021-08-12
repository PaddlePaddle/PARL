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
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(parl.Model):
    """ Neural Network to approximate v value.
    Args:
        obs_dim (int): Dimension of observation space.
        act_dim (int): Dimension of action space.
        softmax (book): Whether to use softmax activation at the end of last layer
    """

    def __init__(self, obs_dim, act_dim, softmax=False):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.softmax = softmax

        self.l1 = nn.Linear(obs_dim, 16)
        self.l2 = nn.Linear(144, 24)
        self.l3 = nn.Linear(40, 128)
        self.l4 = nn.Linear(128, act_dim)

        self.network = nn.Sequential(
            nn.Conv2d(
                in_channels=5, out_channels=16, kernel_size=(3, 3), stride=2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16, out_channels=16, kernel_size=(3, 3), stride=2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(2, 2),
                stride=1,
            ),
            nn.ReLU(),
        )

    def forward(self, x):

        batch_size = x.shape[0]

        ship_feature = x[:, :self.obs_dim]
        world_feature = x[:, self.obs_dim:].reshape((batch_size, 5, 21, 21))
        world_vector = self.network(world_feature).view(batch_size, -1)

        x = F.relu(self.l1(ship_feature))
        y = F.relu(self.l2(world_vector))
        z = F.relu(self.l3(torch.cat((x, y), 1)))
        out = self.l4(z)

        if self.softmax:
            out = F.softmax(out, dim=-1)

        return out
