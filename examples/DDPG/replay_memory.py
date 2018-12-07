#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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


class ReplayMemory(object):
    def __init__(self, size, obs_dim, act_dim):
        self.size = size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.memory = np.zeros((size, obs_dim * 2 + act_dim + 2),
                               dtype=np.float32)
        self.pointer = 0

    def sample_batch(self, batch_size):
        indices = np.random.choice(self.size, size=batch_size)
        data = self.memory[indices, :]
        obs = data[:, :self.obs_dim]
        action = data[:, self.obs_dim:self.obs_dim + self.act_dim]
        reward = data[:, -self.obs_dim - 2:-self.obs_dim - 1]
        next_obs = data[:, -self.obs_dim - 1:-1]
        terminal = data[:, -1:].astype('bool')
        reward = np.squeeze(reward)
        terminal = np.squeeze(terminal)
        return obs, action, reward, next_obs, terminal

    def store_transition(self, obs, action, reward, next_obs, terminal):
        transition = np.hstack((obs, action, [reward], next_obs, [terminal]))
        index = self.pointer % self.size  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1
