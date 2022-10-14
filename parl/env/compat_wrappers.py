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


def get_gym_version():
    version = gym.__version__
    start_flag = 0
    for start_flag in range(len(version)):
        if version[start_flag] == '.':
            break
    start_flag += 1
    for end_flag in range(start_flag, len(version)):
        if version[end_flag] == '.':
            break
    return int(version[start_flag:end_flag])


class CompatWrapper(gym.Wrapper):
    def __init__(self, env):
        """Map action space [-1, 1] of model output to new action space
        [low_bound, high_bound].
        """

        gym.Wrapper.__init__(self, env)
        attr_list = dir(env)
        if hasattr(env, '_max_episode_steps'):
            self._max_episode_steps = int(self.env._max_episode_steps)
        if hasattr(env, '_elapsed_steps'):
            self._elapsed_steps = self.env._elapsed_steps
        self.count_ep_step = 0
        self.ramdom_seed = 0

    def reset(self, **kwargs):
        if get_gym_version() >= 26:
            kwargs['seed'] = self.ramdom_seed
            obs, info = self.env.reset(**kwargs)
        else:
            obs = self.env.reset(**kwargs)
        return obs

    def seed(self, random_seed):
        if get_gym_version() >= 26:
            self.ramdom_seed = random_seed
        else:
            self.env.seed(random_seed)

    def step(self, model_output_act):
        """
        Args:
            model_output_act(np.array): The values must be in in [-1, 1].
        """

        self.count_ep_step += 1
        if get_gym_version() >= 25:
            obs, reward, done, _, info = self.env.step(model_output_act)
        else:
            obs, reward, done, info = self.env.step(model_output_act)
        if self.count_ep_step >= 1000:
            done = True
            self.count_ep_step = 0
        return obs, reward, done, info
