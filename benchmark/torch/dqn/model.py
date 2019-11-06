#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import torch.nn.functional as F

import parl


class AtariModel(parl.Model):
    """CNN network used in TensorPack examples.

    Args:
        input_channel (int): Input channel of states.
        act_dim (int): Dimension of action space.
        algo (str): which ('DQN', 'Double', 'Dueling') model to use.
    """

    def __init__(self, input_channel, act_dim, algo='DQN'):
        super(AtariModel, self).__init__()
        self.conv1 = nn.Conv2d(
            input_channel, 32, kernel_size=8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.algo = algo
        if self.algo == 'Dueling':
            self.fc1_adv = nn.Linear(7744, 512)
            self.fc1_val = nn.Linear(7744, 512)
            self.fc2_adv = nn.Linear(512, act_dim)
            self.fc2_val = nn.Linear(512, 1)
        else:
            self.fc1 = nn.Linear(7744, 512)
            self.fc2 = nn.Linear(512, act_dim)

        self.reset_params()

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        if self.algo == 'Dueling':
            As = self.fc2_adv(F.relu(self.fc1_adv(x)))
            V = self.fc2_val(F.relu(self.fc1_val(x)))
            Q = As + (V - As.mean(dim=1, keepdim=True))
        else:
            Q = self.fc2(F.relu(self.fc1(x)))
        return Q
