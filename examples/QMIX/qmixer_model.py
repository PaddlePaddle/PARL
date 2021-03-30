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

import parl
from parl import layers
import paddle.fluid as fluid


class QMixerModel(parl.Model):
    ''' A hyper-network to generate paramters for mixing model.
    '''

    def __init__(self, config):
        self.n_agents = config['n_agents']
        self.batch_size = config['batch_size']
        self.state_shape = config['state_shape']
        self.embed_dim = config['mixing_embed_dim']
        self.episode_limit = config['episode_limit']
        self.hypernet_layers = config['hypernet_layers']
        self.hypernet_embed_dim = config['hypernet_embed_dim']

        if self.hypernet_layers == 1:
            self.hyper_w_1 = layers.fc(
                size=self.embed_dim * self.n_agents,
                act=None,
                name='hyper_w_1')
            self.hyper_w_2 = layers.fc(
                size=self.embed_dim, act=None, name='hyper_w_2')
        elif self.hypernet_layers == 2:
            self.hyper_w_1_1 = layers.fc(
                size=self.hypernet_embed_dim, name='hyper_w_1_1')
            self.hyper_w_1_2 = layers.fc(
                size=self.embed_dim * self.n_agents, name='hyper_w_1_2')
            self.hyper_w_2_1 = layers.fc(
                size=self.hypernet_embed_dim, name='hyper_w_2_1')
            self.hyper_w_2_2 = layers.fc(
                size=self.embed_dim, name='hyper_w_2_2')
        else:
            raise ValueError('hypernet_layers should be "1" or "2"!')

        self.hyper_b_1 = layers.fc(
            size=self.embed_dim, act=None, name='hyper_b_1')

        self.hyper_b_2_1 = layers.fc(
            size=self.embed_dim, act=None, name='hyper_b_2_1')
        self.hyper_b_2_2 = layers.fc(size=1, act=None, name='hyper_b_2_2')

    def forward(self, agent_qs, states):
        '''
        Args:
            agent_qs: (batch_size, T, n_agents)
            states:   (batch_size, T, state_shape)
        Returns:
            q_total: global q value
        '''
        episode_len = self.episode_limit - 1
        assert agent_qs.shape[1] == episode_len
        states = fluid.layers.reshape(
            states, shape=(self.batch_size * episode_len, self.state_shape))
        agent_qs = fluid.layers.reshape(
            agent_qs, shape=(self.batch_size * episode_len, 1, self.n_agents))

        if self.hypernet_layers == 1:
            w1 = self.hyper_w_1(states)
            w2 = self.hyper_w_2(states)
        elif self.hypernet_layers == 2:
            w1 = self.hyper_w_1_2(fluid.layers.relu(self.hyper_w_1_1(states)))
            w2 = self.hyper_w_2_2(fluid.layers.relu(self.hyper_w_2_1(states)))
        else:
            pass

        w1 = fluid.layers.abs(w1)
        w1 = fluid.layers.reshape(
            w1,
            shape=(self.batch_size * episode_len, self.n_agents,
                   self.embed_dim))
        w2 = fluid.layers.abs(w2)
        w2 = fluid.layers.reshape(
            w2, shape=(self.batch_size * episode_len, self.embed_dim, 1))

        b1 = self.hyper_b_1(states)
        b1 = fluid.layers.reshape(
            b1, shape=(self.batch_size * episode_len, 1, self.embed_dim))

        b2 = self.hyper_b_2_1(states)
        b2 = fluid.layers.relu(b2)
        b2 = self.hyper_b_2_2(b2)
        b2 = fluid.layers.reshape(
            b2, shape=(self.batch_size * episode_len, 1, 1))

        hidden = fluid.layers.elu(fluid.layers.matmul(agent_qs, w1) + b1)
        y = fluid.layers.matmul(hidden, w2) + b2
        q_total = fluid.layers.reshape(
            y, shape=(self.batch_size, episode_len, 1))
        return q_total
