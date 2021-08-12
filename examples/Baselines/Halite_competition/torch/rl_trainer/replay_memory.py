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
    """ Replay Memory for saving data.
    Args:
        max_size (int): size of replay memory
        obs_dim (int): dimension of the observation
    """

    def __init__(self, max_size, obs_dim):

        self.max_size = int(max_size)

        self.obs_dim = obs_dim

        self.reset()

    def sample_batch(self, batch_size):
        if batch_size > self._curr_size:
            batch_idx = np.arange(self._curr_size)
        else:
            batch_idx = np.random.randint(self._curr_size, size=batch_size)

        obs = self.obs[batch_idx]
        action = self.action[batch_idx]
        value = self.value[batch_idx]
        returns = self.returns[batch_idx].reshape((-1, 1))
        log_prob = self.log_prob[batch_idx]
        adv = self.adv[batch_idx]

        return obs, action, value, returns, log_prob, adv

    def make_index(self, batch_size):
        batch_idx = np.random.randint(self._curr_size, size=batch_size)
        return batch_idx

    def sample_batch_by_index(self, batch_idx):
        obs = self.obs[batch_idx]
        action = self.action[batch_idx]
        value = self.value[batch_idx]
        returns = self.returns[batch_idx]
        log_prob = self.log_prob[batch_idx]
        adv = self.adv[batch_idx]

        return obs, action, value, returns, log_prob, adv

    def append(self, obs, act, value, returns, log_prob, adv):

        size = len(obs)

        self._curr_size = min(self._curr_size + size, self.max_size)

        if self._curr_pos + size >= self.max_size:

            delta_size = -(size + self._curr_pos - self.max_size)

            self.obs = np.roll(self.obs, delta_size, 0)
            self.action = np.roll(self.action, delta_size)
            self.value = np.roll(self.value, delta_size)
            self.returns = np.roll(self.returns, delta_size)
            self.log_prob = np.roll(self.log_prob, delta_size)
            self.adv = np.roll(self.adv, delta_size)

            self._curr_pos += delta_size

        self.obs[self._curr_pos:self._curr_pos + size] = obs
        self.action[self._curr_pos:self._curr_pos + size] = act
        self.value[self._curr_pos:self._curr_pos + size] = value
        self.returns[self._curr_pos:self._curr_pos + size] = returns
        self.log_prob[self._curr_pos:self._curr_pos + size] = log_prob
        self.adv[self._curr_pos:self._curr_pos + size] = adv

        self._curr_pos = (self._curr_pos + size) % self.max_size

    def size(self):
        return self._curr_size

    def __len__(self):
        return self._curr_size

    def reset(self):

        self.obs = np.zeros((self.max_size, self.obs_dim), dtype='float32')
        self.action = np.zeros((self.max_size, ), dtype='int32')
        self.value = np.zeros((self.max_size, ), dtype='float32')
        self.returns = np.zeros((self.max_size, ), dtype='float32')
        self.log_prob = np.zeros((self.max_size, ), dtype='float32')
        self.adv = np.zeros((self.max_size, ), dtype='float32')

        self._curr_size = 0
        self._curr_pos = 0
