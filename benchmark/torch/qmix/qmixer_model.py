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

import torch
import torch.nn as nn
import torch.nn.functional as F


class QMixerModel(nn.Module):
    '''
    input: n_agents' agent_qs (a scalar for each agent)
    output: a scalar (Q)
    '''

    def __init__(self, n_agents, state_shape, mixing_embed_dim=64):
        super(QMixerModel, self).__init__()

        self.n_agents = n_agents
        self.state_shape = state_shape
        self.embed_dim = mixing_embed_dim

        self.hyper_w_1 = nn.Linear(self.state_shape,
                                   self.embed_dim * self.n_agents)
        self.hyper_w_2 = nn.Linear(self.state_shape, self.embed_dim)
        self.hyper_b_1 = nn.Linear(self.state_shape, self.embed_dim)
        self.hyper_b_2 = nn.Sequential(
            nn.Linear(self.state_shape, self.embed_dim), nn.ReLU(),
            nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        '''
        Args:
            agent_qs: (batch_size, max_episode_len, n_agents)
            states:
        '''
        batch_size = agent_qs.size(0)
        states = states.reshape(-1,
                                self.state_shape)  # (batch_size, state_shape)
        agent_qs = agent_qs.view(-1, 1,
                                 self.n_agents)  # (batch_size, 1, n_agents)

        w1 = torch.abs(self.hyper_w_1(states))  # keep non-negative
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = self.hyper_b_1(states)
        b1 = b1.view(-1, 1, self.embed_dim)

        w2 = torch.abs(self.hyper_w_2(states))  # keep non-negative
        w2 = w2.view(-1, self.embed_dim, 1)
        b2 = self.hyper_b_2(states).view(-1, 1, 1)

        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        y = torch.bmm(hidden, w2) + b2
        q_total = y.view(batch_size, -1, 1)
        return q_total

    def update(self, model):
        for param, target_param in zip(model.parameters(), self.parameters()):
            target_param.data.copy_(param.data)
