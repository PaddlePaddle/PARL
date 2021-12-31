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

#-*- coding: utf-8 -*-

import parl
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class Model(parl.Model):
    def __init__(self, obs_dim, act_dim):
        super(Model, self).__init__()
        self.actor_model = Actor(obs_dim, act_dim)
        self.critic_model = Critic(obs_dim, act_dim)

    def policy(self, obs):
        return self.actor_model(obs)

    def value(self, obs, action):
        return self.critic_model(obs, action)

    def get_actor_params(self):
        return self.actor_model.parameters()

    def get_critic_params(self):
        return self.critic_model.parameters()


class Actor(parl.Model):
    def __init__(self, obs_dim, act_dim):
        super(Actor, self).__init__()
        hid_size = 100

        self.l1 = nn.Linear(obs_dim, hid_size)
        self.l2 = nn.Linear(hid_size, act_dim)

    def forward(self, obs):
        hid = F.relu(self.l1(obs))
        means = paddle.tanh(self.l2(hid))
        return means


class Critic(parl.Model):
    def __init__(self, obs_dim, act_dim):
        super(Critic, self).__init__()
        hid_size = 100

        self.l1 = nn.Linear(obs_dim + act_dim, hid_size)
        self.l2 = nn.Linear(hid_size, 1)

    def forward(self, obs, act):
        concat = paddle.concat([obs, act], axis=1)
        hid = F.relu(self.l1(concat))
        Q = self.l2(hid)
        return Q
