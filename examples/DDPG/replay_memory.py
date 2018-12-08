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
    def __init__(self, max_size, obs_dim, act_dim):
        self.max_size = max_size
        self.obs_memory = np.zeros((max_size, obs_dim), dtype='float32')
        self.act_memory = np.zeros((max_size, act_dim), dtype='float32')
        self.reward_memory = np.zeros((max_size, ), dtype='float32')
        self.next_obs_memory = np.zeros((max_size, obs_dim), dtype='float32')
        self.terminal_memory = np.zeros((max_size, ), dtype='bool')
        self._curr_size = 0
        self._curr_pos = 0

    def sample_batch(self, batch_size):
        batch_idx = np.random.choice(self._curr_size, size=batch_size)
        obs = self.obs_memory[batch_idx, :]
        act = self.act_memory[batch_idx, :]
        reward = self.reward_memory[batch_idx]
        next_obs = self.next_obs_memory[batch_idx, :]
        terminal = self.terminal_memory[batch_idx]
        return obs, act, reward, next_obs, terminal

    def append(self, obs, act, reward, next_obs, terminal):
        if self._curr_size < self.max_size:
            self._curr_size += 1
        self.obs_memory[self._curr_pos] = obs
        self.act_memory[self._curr_pos] = act
        self.reward_memory[self._curr_pos] = reward
        self.next_obs_memory[self._curr_pos] = next_obs
        self.terminal_memory[self._curr_pos] = terminal
        self._curr_pos = (self._curr_pos + 1) % self.max_size

    def size(self):
        return self._curr_size
