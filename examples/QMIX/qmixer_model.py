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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import parl


class QMixerModel(parl.Model):
    '''
    input: n agents' agent_qs (a scalar for each agent)
    output: a scalar (Q)
    '''

    def __init__(self,
                 n_agents,
                 state_shape,
                 mixing_embed_dim=32,
                 hypernet_layers=2,
                 hypernet_embed_dim=64):
        super(QMixerModel, self).__init__()

        self.n_agents = n_agents
        self.state_shape = state_shape
        self.embed_dim = mixing_embed_dim
        if hypernet_layers == 1:
            self.hyper_w_1 = nn.Linear(self.state_shape,
                                       self.embed_dim * self.n_agents)
            self.hyper_w_2 = nn.Linear(self.state_shape, self.embed_dim)
        elif hypernet_layers == 2:
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.state_shape, hypernet_embed_dim), nn.ReLU(),
                nn.Linear(hypernet_embed_dim, self.embed_dim * self.n_agents))
            self.hyper_w_2 = nn.Sequential(
                nn.Linear(self.state_shape, hypernet_embed_dim), nn.ReLU(),
                nn.Linear(hypernet_embed_dim, self.embed_dim))
        else:
            raise ValueError('hypernet_layers should be "1" or "2"!')

        self.hyper_b_1 = nn.Linear(self.state_shape, self.embed_dim)
        self.hyper_b_2 = nn.Sequential(
            nn.Linear(self.state_shape, self.embed_dim), nn.ReLU(),
            nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        '''
        Args:
            agent_qs (paddle.Tensor): (batch_size, T, n_agents)
            states (paddle.Tensor):   (batch_size, T, state_shape)
        Returns:
            q_total (paddle.Tensor):  (batch_size, T, 1)
        '''
        batch_size = agent_qs.shape[0]
        states = states.reshape(shape=(-1, self.state_shape))
        agent_qs = agent_qs.reshape(shape=(-1, 1, self.n_agents))

        w1 = paddle.abs(self.hyper_w_1(states))
        w1 = w1.reshape(shape=(-1, self.n_agents, self.embed_dim))
        b1 = self.hyper_b_1(states)
        b1 = b1.reshape(shape=(-1, 1, self.embed_dim))

        w2 = paddle.abs(self.hyper_w_2(states))
        w2 = w2.reshape(shape=(-1, self.embed_dim, 1))
        b2 = self.hyper_b_2(states).reshape(shape=(-1, 1, 1))

        hidden = F.elu(paddle.bmm(agent_qs, w1) + b1)
        y = paddle.bmm(hidden, w2) + b2
        q_total = y.reshape(shape=(batch_size, -1, 1))
        return q_total
