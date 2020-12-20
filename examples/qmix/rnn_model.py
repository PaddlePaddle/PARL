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

import numpy as np
import parl
from parl import layers
import paddle.fluid as fluid


class RNNModel(parl.Model):
    ''' GRU-based policy model.
    '''

    def __init__(self, config):
        self.n_actions = config['n_actions']
        self.rnn_hidden_dim = config['rnn_hidden_dim']

        self.fc1 = layers.fc(size=self.rnn_hidden_dim, act=None, name='fc1')
        self.gru = layers.GRUCell(hidden_size=self.rnn_hidden_dim, name='gru')
        self.fc2 = layers.fc(size=self.n_actions, act=None, name='fc2')

    def __call__(self, inputs, hidden_state):
        """
        Args:
            inputs:       (batch_size * n_agents, rnn_hidden_dim)
            hidden_state: (batch_size, rnn_hidden_dim)
        Returns:
            q: local q values
            h: hidden states
        """
        x = fluid.layers.relu(self.fc1(inputs))
        h, _ = self.gru(x, hidden_state)
        q = self.fc2(h)
        return q, h
