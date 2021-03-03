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
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

# clamp bounds for Std of action_log
LOG_SIG_MAX = 2.0
LOG_SIG_MIN = -20.0


class CarlaModel(parl.Model):
    def __init__(self, obs_dim, action_dim):
        super(CarlaModel, self).__init__()
        self.actor_model = Actor(obs_dim, action_dim)
        self.critic_model = Critic(obs_dim, action_dim)

    def policy(self, obs):
        return self.actor_model(obs)

    def value(self, obs, action):
        return self.critic_model(obs, action)

    def get_actor_params(self):
        return self.actor_model.parameters()

    def get_critic_params(self):
        return self.critic_model.parameters()


class Actor(parl.Model):
    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(obs_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)

        self.mean_linear1 = nn.Linear(256, 256)
        self.mean_linear2 = nn.Linear(256, 256)
        self.mean_linear = nn.Linear(256, action_dim)

        self.std_linear1 = nn.Linear(256, 256)
        self.std_linear2 = nn.Linear(256, 256)
        self.std_linear = nn.Linear(256, action_dim)

    def forward(self, obs):
        x = F.relu(self.l1(obs))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))

        act_mean = F.relu(self.mean_linear1(x))
        act_mean = F.relu(self.mean_linear2(act_mean))
        act_mean = self.mean_linear(act_mean)

        act_std = F.relu(self.std_linear1(x))
        act_std = F.relu(self.std_linear2(act_std))
        act_std = self.std_linear(act_std)
        act_log_std = paddle.clip(act_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return act_mean, act_log_std


class Critic(parl.Model):
    def __init__(self, obs_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 network
        self.l1 = nn.Linear(obs_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, 256)
        self.l5 = nn.Linear(256, 1)

        # Q2 network
        self.l6 = nn.Linear(obs_dim + action_dim, 256)
        self.l7 = nn.Linear(256, 256)
        self.l8 = nn.Linear(256, 256)
        self.l9 = nn.Linear(256, 256)
        self.l10 = nn.Linear(256, 1)

    def forward(self, obs, action):
        x = paddle.concat([obs, action], 1)

        # Q1
        q1 = F.relu(self.l1(x))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = F.relu(self.l4(q1))
        q1 = self.l5(q1)

        # Q2
        q2 = F.relu(self.l6(x))
        q2 = F.relu(self.l7(q2))
        q2 = F.relu(self.l8(q2))
        q2 = F.relu(self.l9(q2))
        q2 = self.l10(q2)
        return q1, q2
