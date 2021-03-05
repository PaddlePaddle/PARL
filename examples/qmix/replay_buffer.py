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

from collections import deque
import numpy as np
import random


class EpisodeExperience(object):
    def __init__(self, episode_len):
        self.max_len = episode_len

        self.episode_state = []
        self.episode_actions = []
        self.episode_reward = []
        self.episode_terminated = []
        self.episode_obs = []
        self.episode_available_actions = []
        self.episode_filled = []

    @property
    def count(self):
        return len(self.episode_state)

    def add(self, state, actions, reward, terminated, obs, available_actions,
            filled):
        assert self.count < self.max_len
        self.episode_state.append(state)
        self.episode_actions.append(actions)
        self.episode_reward.append(reward)
        self.episode_terminated.append(terminated)
        self.episode_obs.append(obs)
        self.episode_available_actions.append(available_actions)
        self.episode_filled.append(filled)

    def get_data(self):
        assert self.count == self.max_len
        return np.array(self.episode_state), np.array(self.episode_actions),\
                np.array(self.episode_reward), np.array(self.episode_terminated),\
                np.array(self.episode_obs),\
                np.array(self.episode_available_actions), np.array(self.episode_filled)


class EpisodeReplayBuffer(object):
    def __init__(self, max_buffer_size):
        self.max_buffer_size = max_buffer_size
        self.buffer = deque(maxlen=max_buffer_size)

    def add(self, episode_experience):
        self.buffer.append(episode_experience)

    @property
    def count(self):
        return len(self.buffer)

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)
        s_batch, a_batch, r_batch, t_batch, obs_batch, available_actions_batch,\
                filled_batch = [], [], [], [], [], [], []
        for episode in batch:
            s, a, r, t, obs, available_actions, filled = episode.get_data()
            s_batch.append(s)
            a_batch.append(a)
            r_batch.append(r)
            t_batch.append(t)
            obs_batch.append(obs)
            available_actions_batch.append(available_actions)
            filled_batch.append(filled)

        s_batch = np.array(s_batch, dtype='float32')
        filled_batch = np.array(filled_batch, dtype='float32')
        r_batch = np.array(r_batch, dtype='float32')
        t_batch = np.array(t_batch, dtype='float32')
        a_batch = np.array(a_batch, dtype='long')
        obs_batch = np.array(obs_batch, dtype='float32')
        available_actions_batch = np.array(
            available_actions_batch, dtype='long')

        return s_batch, a_batch, r_batch, t_batch, obs_batch,\
                available_actions_batch, filled_batch
