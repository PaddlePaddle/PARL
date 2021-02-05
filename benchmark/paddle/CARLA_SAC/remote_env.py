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
from parl.utils import logger, tensorboard
from parl.env.continuous_wrappers import ActionMappingWrapper


class ParallelEnv(object):
    def __init__(self, env_name, xparl_addr, train_envs_params):
        parl.connect(xparl_addr)
        self.env_list = [
            CarlaRemoteEnv(env_name=env_name, params=params)
            for params in train_envs_params
        ]
        self.episode_reward_list = [0] * len(self.env_list)
        self.episode_steps_list = [0] * len(self.env_list)
        self._max_episode_steps = train_envs_params[0]['max_time_episode']
        self.total_steps = 0

    def reset(self):
        obs_list = [env.reset() for env in self.env_list]
        obs_list = [obs.get() for obs in obs_list]
        self.obs_list = np.array(obs_list)
        return self.obs_list

    def step(self, action_list):
        return_list = [
            self.env_list[i].step(action_list[i])
            for i in range(len(self.env_list))
        ]
        return_list = [return_.get() for return_ in return_list]
        return_list = np.array(return_list, dtype=object)
        self.next_obs_list = return_list[:, 0]
        self.reward_list = return_list[:, 1]
        self.done_list = return_list[:, 2]
        self.info_list = return_list[:, 3]
        return self.next_obs_list, self.reward_list, self.done_list, self.info_list

    def get_obs(self):
        for i in range(len(self.env_list)):
            self.total_steps += 1
            self.episode_steps_list[i] += 1
            self.episode_reward_list[i] += self.reward_list[i]

            self.obs_list[i] = self.next_obs_list[i]
            if self.done_list[i] or self.episode_steps_list[
                    i] >= self._max_episode_steps:
                tensorboard.add_scalar('train/episode_reward_env{}'.format(i),
                                       self.episode_reward_list[i],
                                       self.total_steps)
                logger.info('Train env {} done, Reward: {}'.format(
                    i, self.episode_reward_list[i]))

                self.episode_steps_list[i] = 0
                self.episode_reward_list[i] = 0
                obs_list_i = self.env_list[i].reset()
                self.obs_list[i] = obs_list_i.get()
                self.obs_list[i] = np.array(self.obs_list[i])
        return self.obs_list


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

    def seed(self, seed):
        return self.env.seed(seed)


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
        self.env.seed(params['port'])
        self._max_episode_steps = int(params['max_time_episode'])
        self.action_space = ActionSpace(
            self.env.action_space, self.env.action_space.low,
            self.env.action_space.high, self.env.action_space.shape)

    def reset(self):
        obs, _ = self.env.reset()
        return obs

    def step(self, action):
        return self.env.step(action)

    def seed(self, seed):
        return self.env.seed(seed)
