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
import operator

# V_GYM_CHANGED change env.seed(), env.reset() and env.step()
# V_NPRANDOM_CHANGED change env.np_random from .randint() to .integers()
V_GYM_CHANGED = '0.26.0'
V_NPRANDOM_CHANGED = '0.22.0'

__all__ = [
    'CompatWrapper', 'is_gym_version_ge', 'V_GYM_CHANGED', 'V_NPRANDOM_CHANGED'
]


def is_gym_version_ge(compare_version):
    # Check whether the version of gym is greater than compare_version
    # get gym version in python env
    version_num = gym.__version__.split('.')
    for i in range(len(version_num)):
        try:
            version_num[i] = int(version_num[i])
        except:
            pass
    # deal with compare_version
    compare_num = compare_version.split('.')
    for i in range(len(version_num)):
        try:
            compare_num[i] = int(compare_num[i])
        except:
            pass
    return operator.ge(version_num, compare_num)


class CompatWrapper(gym.Wrapper):
    """ Compatible for different versions of gym, especially for `step()` and `reset()` APIs.

        .. code-block:: python
        # old version (< 0.26.0) of gym APIs
        observation = env.reset()
        observation, reward, done, info = env.step(action)
        # new version (>= 0.26.0) of gym APIs
        observation, info = env.reset()
        observation, reward, terminated, truncated, info = env.step(action)
    
    After being wrapped by `CompatWrapper`, the new version of the gym env can be used in the same way as the old version of gym.
    """

    def __init__(self, env):
        super().__init__(env)
        if hasattr(env, '_max_episode_steps'):
            self._max_episode_steps = int(self.env._max_episode_steps)
        if hasattr(env, '_elapsed_steps'):
            self._elapsed_steps = self.env._elapsed_steps
        self.count_ep_step = 0
        self.random_seed = 'without_setting'

    def reset(self, **kwargs):
        if is_gym_version_ge(V_GYM_CHANGED):
            if self.random_seed != "without_setting":
                kwargs['seed'] = self.random_seed
            obs, info = self.env.reset(**kwargs)
        else:
            obs = self.env.reset(**kwargs)
        return obs

    def seed(self, random_seed):
        if is_gym_version_ge(V_GYM_CHANGED):
            self.random_seed = random_seed
        else:
            self.env.seed(random_seed)

    def step(self, action):
        self.count_ep_step += 1
        if is_gym_version_ge(V_GYM_CHANGED):
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
