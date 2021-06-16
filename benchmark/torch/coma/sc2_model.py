#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import parl


class ComaModel(parl.Model):
    def __init__(self, config):
        super(ComaModel, self).__init__()
        self.n_actions = config['n_actions']
        self.n_agents = config['n_agents']
        self.state_shape = config['state_shape']
        self.obs_shape = config['obs_shape']

        actor_input_dim = self._get_actor_input_dim()
        critic_input_dim = self._get_critic_input_dim()

        self.actor_model = ActorModel(actor_input_dim, self.n_actions)
        self.critic_model = CriticModel(critic_input_dim, self.n_actions)

    def policy(self, obs, hidden_state):
        return self.actor_model(obs, hidden_state)

    def value(self, inputs):
        return self.critic_model(inputs)

    def get_actor_params(self):
        return self.actor_model.parameters()

    def get_critic_params(self):
        return self.critic_model.parameters()

    def _get_actor_input_dim(self):
        input_shape = self.obs_shape  # obs: 30 in 3m map
        input_shape += self.n_actions  # agent's last action (one_hot): 9 in 3m map
        input_shape += self.n_agents  # agent's one_hot id: 3 in 3m map
        return input_shape  # 30 + 9 + 3 = 42

    def _get_critic_input_dim(self):
        input_shape = self.state_shape  # state: 48 in 3m map
        input_shape += self.obs_shape  # obs: 30 in 3m map
        input_shape += self.n_agents  # agent_id: 3 in 3m map
        input_shape += self.n_actions * self.n_agents * 2  # all agents' action and last_action (one-hot): 54 in 3m map
        return input_shape  # 48 + 30+ 3 = 135


# all agents share one actor network
class ActorModel(parl.Model):
    def __init__(self, input_shape, act_dim):
        """ input : obs, include the agent's id and last action, shape: (batch, obs_shape + n_action + n_agents)
            output: one agent's q(obs, act)
        """
        super(ActorModel, self).__init__()
        self.hid_size = 64

        self.fc1 = nn.Linear(input_shape, self.hid_size)
        self.rnn = nn.GRUCell(self.hid_size, self.hid_size)
        self.fc2 = nn.Linear(self.hid_size, act_dim)

    def init_hidden(self):
        # new hidden states
        return self.fc1.weight.new(1, self.hid_size).zero_()

    def forward(self, obs, h0):
        x = F.relu(self.fc1(obs))
        h1 = h0.reshape(-1, self.hid_size)
        h2 = self.rnn(x, h1)
        policy = self.fc2(h2)
        return policy, h2


class CriticModel(parl.Model):
    def __init__(self, input_shape, act_dim):
        """ inputs: [ s(t), o(t)_a, u(t)_a, agent_a, u(t-1) ], shape: (Batch, input_shape)
            output: Q,   shape: (Batch, n_actions)
            Batch = ep_num * n_agents
        """
        super(CriticModel, self).__init__()
        hid_size = 128
        self.fc1 = nn.Linear(input_shape, hid_size)
        self.fc2 = nn.Linear(hid_size, hid_size)
        self.fc3 = nn.Linear(hid_size, act_dim)

    def forward(self, inputs):
        hid1 = F.relu(self.fc1(inputs))
        hid2 = F.relu(self.fc2(hid1))
        Q = self.fc3(hid2)
        return Q
