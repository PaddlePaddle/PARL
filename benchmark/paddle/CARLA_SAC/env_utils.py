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

import parl
import carla
import gym
import gym_carla
import numpy as np
from parl.env.continuous_wrappers import ActionMappingWrapper


class ParallelEnv(object):
    def __init__(self, env_name, train_envs_params):
        self.env_list = [
            CarlaRemoteEnv(env_name=env_name, params=params)
            for params in train_envs_params
        ]
        self.env_num = len(self.env_list)

        self.episode_steps_list = [0] * self.env_num
        self._max_episode_steps = train_envs_params[0]['max_time_episode']

    def reset(self):
        obs_list = [env.reset() for env in self.env_list]
        obs_list = [obs.get() for obs in obs_list]
        obs_list = np.array(obs_list)
        return obs_list

    def step(self, action_list):
        return_list = [
            self.env_list[i].step(action_list[i]) for i in range(self.env_num)
        ]
        return_list = [return_.get() for return_ in return_list]
        return_list = np.array(return_list, dtype=object)
        next_obs_list = return_list[:, 0]
        reward_list = return_list[:, 1]
        done_list = return_list[:, 2]
        info_list = return_list[:, 3]

        for i in range(self.env_num):
            self.episode_steps_list[i] += 1
            info_list[i]['timeout'] = False

            if done_list[i] or self.episode_steps_list[
                    i] >= self._max_episode_steps:
                if self.episode_steps_list[i] >= self._max_episode_steps:
                    info_list[i]['timeout'] = True
                self.episode_steps_list[i] = 0
                obs_list_i = self.env_list[i].reset()
                next_obs_list[i] = obs_list_i.get()
                next_obs_list[i] = np.array(next_obs_list[i])
        return next_obs_list, reward_list, done_list, info_list


class LocalEnv(object):
    def __init__(self, env_name, params):
        self.env = gym.make(env_name, params=params)
        self.env = ActionMappingWrapper(self.env)
        self._max_episode_steps = int(params['max_time_episode'])
        self.obs_dim = self.env.state_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

    def reset(self):
        obs, _ = self.env.reset()
        return obs

    def step(self, action):
        return self.env.step(action)


@parl.remote_class(wait=False)
class CarlaRemoteEnv(object):
    def __init__(self, env_name, params):
        class ActionSpace(object):
            def __init__(self,
                         action_space=None,
                         low=None,
                         high=None,
                         shape=None,
                         n=None):
                self.action_space = action_space
                self.low = low
                self.high = high
                self.shape = shape
                self.n = n

            def sample(self):
                return self.action_space.sample()

        self.env = gym.make(env_name, params=params)
        self.env = ActionMappingWrapper(self.env)
        self._max_episode_steps = int(params['max_time_episode'])
        self.action_space = ActionSpace(
            self.env.action_space, self.env.action_space.low,
            self.env.action_space.high, self.env.action_space.shape)

    def reset(self):
        obs, _ = self.env.reset()
        return obs

    def step(self, action):
        return self.env.step(action)
