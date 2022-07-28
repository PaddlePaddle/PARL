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
import numpy as np


class PPOModel(parl.Model):
    def __init__(self, obs_space, act_space):
        super().__init__()

        self.continuous_action = True
        if hasattr(act_space, 'high'):
            self.actor_critic = MujocoModel(obs_space, act_space)
        elif hasattr(act_space, 'n'):
            self.continuous_action = False
            self.actor_critic = AtariModel(obs_space, act_space)
        else:
            raise AssertionError("act_space must be instance of gym.spaces.Box or gym.spaces.Discrete")
    
    def value(self, obs):
        return self.actor_critic.value(obs)
    
    def policy(self, obs):
        if self.continuous_action:
            action_mean, action_std = self.actor_critic.policy(obs)
            return action_mean, action_std
        else:
            logits = self.actor_critic.policy(obs)
            return logits


def _init_layer(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class AtariModel(parl.Model):
    def __init__(self, obs_space, act_space):
        super(AtariModel, self).__init__()
        
        self.conv1 = _init_layer(nn.Conv2d(4, 32, 8, stride=4))
        self.conv2 = _init_layer(nn.Conv2d(32, 64, 4, stride=2))
        self.conv3 = _init_layer(nn.Conv2d(64, 64, 3, stride=1))

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = _init_layer(nn.Linear(64 * 7 * 7, 512))

        self.fc_pi = _init_layer(nn.Linear(512, act_space.n), std=0.01)
        self.fc_v = _init_layer(nn.Linear(512, 1), std=1)

    def value(self, obs):
        obs = obs / 255.0
        out = self.relu(self.conv1(obs))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))

        out = self.fc(self.flatten(out))
        value = self.fc_v(out)
        return value
    
    def policy(self, obs):
        obs = obs / 255.0
        out = self.relu(self.conv1(obs))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))

        out = self.fc(self.flatten(out))
        logits = self.fc_pi(out)
        return logits


class MujocoModel(parl.Model):
    def __init__(self, obs_space, act_space):
        super(MujocoModel, self).__init__()

        self.fc_v1 = _init_layer(nn.Linear(obs_space.shape[0], 64))
        self.fc_v2 = _init_layer(nn.Linear(64, 64))
        self.fc_v3 = _init_layer(nn.Linear(64, 1), std=1.0)

        self.fc_pi1 = _init_layer(nn.Linear(obs_space.shape[0], 64))
        self.fc_pi2 = _init_layer(nn.Linear(64, 64))
        self.fc_pi3 = _init_layer(nn.Linear(64, np.prod(act_space.shape)), std=0.01)

        self.tanh = nn.Tanh()
        self.fc_pi_std = nn.Parameter(torch.zeros(1, act_space.shape[0]))

    def value(self, obs):
        out = self.tanh(self.fc_v1(obs))
        out = self.tanh(self.fc_v2(out))
        value = self.fc_v3(out)
        return value
    
    def policy(self, obs):
        out = self.tanh(self.fc_pi1(obs))
        out = self.tanh(self.fc_pi2(out))
        action_mean = self.fc_pi3(out)

        action_logstd = self.fc_pi_std.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        return action_mean, action_std
