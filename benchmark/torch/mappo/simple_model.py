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
import torch.nn as nn
import torch.nn.functional as F


class SimpleModel(parl.Model):
    def __init__(self, obs_dim, cent_obs_dim, act_dim):
        super(SimpleModel, self).__init__()
        self.act_dim = act_dim

        self.actor = Actor(obs_dim, self.act_dim)
        self.critic = Critic(cent_obs_dim)

    def policy(self, obs):
        actions = self.actor(obs)
        return actions

    def value(self, cent_obs):
        values = self.critic(cent_obs)
        return values


class Actor(parl.Model):
    def __init__(self, obs_dim, act_dim):
        super(Actor, self).__init__()
        self.multi_discrete = False
        self.ln1 = nn.LayerNorm(obs_dim)
        self.ln2 = nn.LayerNorm(64)
        self.ln3 = nn.LayerNorm(64)
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)

        if isinstance(act_dim, int):
            self.fc3 = nn.Linear(64, act_dim)
        else:
            self.multi_discrete = True
            self.action_outs = []
            for action_dim in act_dim:
                self.action_outs.append(nn.Linear(64, action_dim))
            self.action_outs = nn.ModuleList(self.action_outs)

    def forward(self, obs):
        x = self.ln1(obs)
        x = F.tanh(self.fc1(x))
        x = self.ln2(x)
        x = F.tanh(self.fc2(x))

        if self.multi_discrete:
            policys = []
            for action_out in self.action_outs:
                policy = action_out(x)
                policys.append(policy)
        else:
            policys = self.fc3(x)
        return policys


class Critic(parl.Model):
    def __init__(self, cent_obs_dim):
        super(Critic, self).__init__()
        self.ln1 = nn.LayerNorm(cent_obs_dim)
        self.ln2 = nn.LayerNorm(64)
        self.ln3 = nn.LayerNorm(64)
        self.fc1 = nn.Linear(cent_obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.v_out = nn.Linear(64, 1)

    def forward(self, cent_obs):
        x = self.ln1(cent_obs)
        x = F.tanh(self.fc1(x))
        x = self.ln2(x)
        x = F.tanh(self.fc2(x))
        values = self.v_out(x)

        return values
