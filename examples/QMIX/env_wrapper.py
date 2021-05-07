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

from utils import OneHotTransform
from copy import deepcopy
import numpy as np


class SC2EnvWrapper(object):
    def __init__(self, env):
        self.env = env
        env_info = env.get_env_info()
        self.episode_limit = env_info['episode_limit']
        self.n_actions = env_info['n_actions']
        self.n_agents = env_info['n_agents']
        self.state_shape = env_info['state_shape']
        self.obs_shape = env_info['obs_shape'] + self.n_agents + self.n_actions
        self.agent_id_one_hot_transform = OneHotTransform(self.n_agents)
        self.actions_one_hot_transform = OneHotTransform(self.n_actions)
        self._init_agents_id_one_hot(self.n_agents)

    @property
    def win_counted(self):
        return self.env.win_counted

    def _init_agents_id_one_hot(self, n_agents):
        agents_id_one_hot = []
        for agent_id in range(self.n_agents):
            one_hot = self.agent_id_one_hot_transform(agent_id)
            agents_id_one_hot.append(one_hot)
        self.agents_id_one_hot = np.array(agents_id_one_hot)

    def _get_agents_id_one_hot(self):
        return deepcopy(self.agents_id_one_hot)

    def _get_actions_one_hot(self, actions):
        actions_one_hot = []
        for action in actions:
            one_hot = self.actions_one_hot_transform(action)
            actions_one_hot.append(one_hot)
        return np.array(actions_one_hot)

    def get_available_actions(self):
        available_actions = []
        for agent_id in range(self.n_agents):
            available_actions.append(
                self.env.get_avail_agent_actions(agent_id))
        return np.array(available_actions)

    def reset(self):
        self.env.reset()
        # action at last timestep
        last_actions_one_hot = np.zeros((self.n_agents, self.n_actions),
                                        dtype='float32')

        obs = np.array(self.env.get_obs())
        agents_id_one_hot = self._get_agents_id_one_hot()
        obs = np.concatenate([obs, last_actions_one_hot, agents_id_one_hot],
                             axis=-1)
        state = np.array(self.env.get_state())
        return state, obs

    def step(self, actions):
        reward, terminated, _ = self.env.step(actions)

        next_state = np.array(self.env.get_state())
        last_actions_one_hot = self._get_actions_one_hot(actions)
        next_obs = np.array(self.env.get_obs())
        next_obs = np.concatenate(
            [next_obs, last_actions_one_hot, self.agents_id_one_hot], axis=-1)
        return next_state, next_obs, reward, terminated
