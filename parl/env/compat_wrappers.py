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

import gym
import numpy as np
from parl.utils import logger

# BASE_VERSION1 make some changes in env.seed() and env.reset()
# BASE_VERSION2 make some changes in change env.step()
BASE_VERSION1 = [0, 26, 0]
BASE_VERSION2 = [0, 25, 0]


def get_gym_version():
    version = gym.__version__
    version_num = [int(x) for x in version.split('.')]
    return version_num


def compare_version(current_version_num, base_version_num):
    if len(current_version_num) != len(base_version_num):
        raise ValueError('Version format must be x.x.x')
    else:
        for i in range(len(current_version_num)):
            if current_version_num[i] < base_version_num[i]:
                return "Low"
            elif current_version_num[i] > base_version_num[i]:
                return 'HighEqual'
    return 'HighEqual'


class CompatWrapper(gym.Wrapper):
    def __init__(self, env):
        """
        Compat mujoco-v4
        """

        super().__init__(env)
        attr_list = dir(env)
        if hasattr(env, '_max_episode_steps'):
            self._max_episode_steps = int(self.env._max_episode_steps)
        if hasattr(env, '_elapsed_steps'):
            self._elapsed_steps = self.env._elapsed_steps
        self.count_ep_step = 0
        self.ramdom_seed = 'without_setting'

    def reset(self, **kwargs):
        if compare_version(get_gym_version(), BASE_VERSION1) == "HighEqual":
            if self.ramdom_seed != "without_setting":
                kwargs['seed'] = self.ramdom_seed
            obs, info = self.env.reset(**kwargs)
        else:
            obs = self.env.reset(**kwargs)
        return obs

    def seed(self, random_seed):
        if compare_version(get_gym_version(), BASE_VERSION1) == "HighEqual":
            self.ramdom_seed = random_seed
        else:
            self.env.seed(random_seed)

    def step(self, action):
        self.count_ep_step += 1
        if compare_version(get_gym_version(), BASE_VERSION2) == "HighEqual":
            obs, reward, done, _, info = self.env.step(action)
            if hasattr(self.env, '_elapsed_steps'):
                self._elapsed_steps = self.env._elapsed_steps
        else:
            obs, reward, done, info = self.env.step(action)
            if hasattr(env, '_elapsed_steps'):
                self._elapsed_steps = self.env._elapsed_steps
        if self.count_ep_step >= self._max_episode_steps:
            done = True
            self.count_ep_step = 0
        return obs, reward, done, info
