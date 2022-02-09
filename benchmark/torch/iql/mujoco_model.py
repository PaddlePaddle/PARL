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

from cmath import log
import parl
import torch
import torch.nn as nn
import torch.nn.functional as F

# clamp bounds for Std of action_log
LOG_SIG_MAX = 0.0
LOG_SIG_MIN = -6


class MujocoModel(parl.Model):
    def __init__(self, obs_dim, action_dim):
        super(MujocoModel, self).__init__()
        self.actor_model = Actor(obs_dim, action_dim)
        self.critic_model = Critic(obs_dim, action_dim)
        self.value_model = Value(obs_dim)

    def policy(self, obs):
        return self.actor_model(obs)

    def qvalue(self, obs, action):
        return self.critic_model(obs, action)

    def value(self, obs):
        return self.value_model(obs)

    def get_actor_params(self):
        return self.actor_model.parameters()

    def get_critic_params(self):
        return self.critic_model.parameters()

    def get_value_params(self):
        return self.value_model.parameters()


class Value(parl.Model):
    def __init__(self, obs_dim):
        super(Value, self).__init__()
        self.l1 = nn.Linear(obs_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.last_fc1 = nn.Linear(256, 1)

    def forward(self, obs):
        x = obs
        v = F.relu(self.l1(x))
        v = F.relu(self.l2(v))
        v = self.last_fc1(v)
        return v


class Critic(parl.Model):
    def __init__(self, obs_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 network
        self.l1 = nn.Linear(obs_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.last_fc1 = nn.Linear(256, 1)

        # Q2 network
        self.l4 = nn.Linear(obs_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.last_fc2 = nn.Linear(256, 1)

    def forward(self, obs, action):
        x = torch.cat([obs, action], 1)
        # Q1
        q1 = F.relu(self.l1(x))
        q1 = F.relu(self.l2(q1))
        q1 = self.last_fc1(q1)
        # Q2
        q2 = F.relu(self.l4(x))
        q2 = F.relu(self.l5(q2))
        q2 = self.last_fc2(q2)
        return q1, q2


class Actor(parl.Model):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2):
        super().__init__()

        self.l1 = nn.Linear(obs_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.mean_linear = nn.Linear(256, act_dim)
        self.log_std = nn.Parameter(
            torch.zeros(act_dim, dtype=torch.float32), requires_grad=True)

    def forward(self, obs):
        x = F.relu(self.l1(obs))
        x = F.relu(self.l2(x))
        mean = F.tanh(self.mean_linear(x))
        log_std = torch.sigmoid(self.log_std)
        log_std = LOG_SIG_MIN + log_std * (LOG_SIG_MAX - LOG_SIG_MIN)
        log_std = torch.exp(log_std)
        scale_tril = torch.diag(log_std)
        return torch.distributions.MultivariateNormal(
            mean, scale_tril=scale_tril)

    def act(self, obs, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            dist = self(obs)
            return dist.mean if deterministic else dist.sample()
