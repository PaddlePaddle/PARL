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

# modified from https://github.com/marlbenchmark/on-policy

import parl
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from mappo_buffer import get_shape_from_obs_space


def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output


class MAPPOModel(parl.Model):
    def __init__(self,
                 obs_space,
                 cent_obs_space,
                 act_space,
                 device=torch.device("cpu")):
        super(MAPPOModel, self).__init__()
        self.device = device
        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = Actor(self.obs_space, self.act_space, self.device)
        self.critic = Critic(self.share_obs_space, self.device)

    def policy(self, obs, available_actions=None, deterministic=False):
        actions = self.actor(obs, available_actions, deterministic)
        return actions

    def value(self, cent_obs):
        values = self.critic(cent_obs)
        return values


class Actor(parl.Model):
    def __init__(self, obs_space, action_space, device=torch.device("cpu")):
        super(Actor, self).__init__()
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.multi_discrete = False
        obs_shape = get_shape_from_obs_space(obs_space)
        self.ln1 = nn.LayerNorm(obs_shape[0])
        self.ln2 = nn.LayerNorm(64)
        self.ln3 = nn.LayerNorm(64)
        self.fc1 = nn.Linear(obs_shape[0], 64)
        self.fc2 = nn.Linear(64, 64)
        if action_space.__class__.__name__ == "Discrete":
            self.fc3 = nn.Linear(64, action_space.n)
        else:
            self.multi_discrete = True
            action_dims = action_space.high - action_space.low + 1
            self.action_outs = []
            for action_dim in action_dims:
                self.action_outs.append(nn.Linear(64, action_dim))
            self.action_outs = nn.ModuleList(self.action_outs)
        self.to(device)

    def forward(self, obs, available_actions=None, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
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
            if available_actions is not None:
                policys[available_actions == 0] = -1e10

        return policys


class Critic(parl.Model):
    def __init__(self, cent_obs_space, device=torch.device("cpu")):
        super(Critic, self).__init__()
        self.tpdv = dict(dtype=torch.float32, device=device)
        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        self.ln1 = nn.LayerNorm(cent_obs_shape[0])
        self.ln2 = nn.LayerNorm(64)
        self.ln3 = nn.LayerNorm(64)
        self.fc1 = nn.Linear(cent_obs_shape[0], 64)
        self.fc2 = nn.Linear(64, 64)
        self.v_out = nn.Linear(64, 1)
        self.to(device)

    def forward(self, cent_obs):
        cent_obs = check(cent_obs).to(**self.tpdv)
        x = self.ln1(cent_obs)
        x = F.tanh(self.fc1(x))
        x = self.ln2(x)
        x = F.tanh(self.fc2(x))
        values = self.v_out(x)

        return values
