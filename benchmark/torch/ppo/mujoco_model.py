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

import parl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class MujocoModel(parl.Model):
    def __init__(self, obs_dim, act_dim):
        super(MujocoModel, self).__init__()
        self.actor = Actor(obs_dim, act_dim)
        self.critic = Critic(obs_dim)

    def policy(self, obs):
        return self.actor(obs)

    def value(self, obs):
        return self.critic(obs)


class Actor(parl.Model):
    def __init__(self, obs_dim, act_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)

        self.fc_mean = nn.Linear(64, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        x = torch.tanh(self.fc1(obs))
        x = torch.tanh(self.fc2(x))

        mean = self.fc_mean(x)
        return mean, self.log_std


class Critic(parl.Model):
    def __init__(self, obs_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, obs):
        x = torch.tanh(self.fc1(obs))
        x = torch.tanh(self.fc2(x))
        value = self.fc3(x)

        return value
