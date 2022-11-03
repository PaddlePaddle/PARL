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
import operator

# V_RESET_CHANGED change env.seed() and env.reset()
# V_STEP_CHANGED change env.step()
# V_NPRANDOM_CHANGED change env.np_random from .randint() to .integers()
V_RESET_CHANGED = '0.26.0'
V_STEP_CHANGED = '0.25.0'
V_NPRANDOM_CHANGED = '0.21.0'


def get_gym_version(version_str=gym.__version__):
    version_num = version_str.split('.')
    for i in range(len(version_num)):
        try:
            version_num[i] = int(version_num[i])
        except:
            pass
    return version_num


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
        self.random_seed = 'without_setting'

    def reset(self, **kwargs):
        if operator.ge(get_gym_version(), get_gym_version(V_RESET_CHANGED)):
            if self.random_seed != "without_setting":
                kwargs['seed'] = self.random_seed
            obs, info = self.env.reset(**kwargs)
        else:
            obs = self.env.reset(**kwargs)
        return obs

    def seed(self, random_seed):
        if operator.ge(get_gym_version(), get_gym_version(V_RESET_CHANGED)):
            self.random_seed = random_seed
        else:
            self.env.seed(random_seed)

    def step(self, action):
        self.count_ep_step += 1
        if operator.ge(get_gym_version(), get_gym_version(V_STEP_CHANGED)):
            obs, reward, done, _, info = self.env.step(action)
            if hasattr(self.env, '_elapsed_steps'):
                self._elapsed_steps = self.env._elapsed_steps
        else:
            obs, reward, done, info = self.env.step(action)
            if hasattr(self.env, '_elapsed_steps'):
                self._elapsed_steps = self.env._elapsed_steps
        if hasattr(self, '_max_episode_steps') and \
                self.count_ep_step >= self._max_episode_steps:
            done = True
            self.count_ep_step = 0
        return obs, reward, done, info
