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
import torch
import torch.nn as nn
import numpy as np


def _init_layer(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class MujocoModel(parl.Model):
    def __init__(self, obs_space, act_space):
        super(MujocoModel, self).__init__()

        self.fc1 = _init_layer(nn.Linear(obs_space.shape[0], 64))
        self.fc2 = _init_layer(nn.Linear(64, 64))

        self.fc_value = _init_layer(nn.Linear(64, 1), std=1.0)

        self.fc_policy = _init_layer(
            nn.Linear(64, np.prod(act_space.shape)), std=0.01)
        self.fc_pi_std = nn.Parameter(torch.zeros(1, act_space.shape[0]))

    def value(self, obs):
        out = torch.tanh(self.fc1(obs))
        out = torch.tanh(self.fc2(out))
        value = self.fc_value(out)
        return value

    def policy(self, obs):
        out = torch.tanh(self.fc1(obs))
        out = torch.tanh(self.fc2(out))
        action_mean = self.fc_policy(out)

        action_logstd = self.fc_pi_std.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        return action_mean, action_std
