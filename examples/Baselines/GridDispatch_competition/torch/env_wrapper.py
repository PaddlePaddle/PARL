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

import gym
import numpy as np
from parl.utils import logger
from Environment.base_env import Environment
from utilize.settings import settings
from utilize.form_action import *


class MaxTimestepWrapper(gym.Wrapper):
    def __init__(self, env, max_timestep=288):
        logger.info("[env type]:{}".format(type(env)))
        self.max_timestep = max_timestep
        env.observation_space = None
        env.reward_range = None
        env.metadata = None
        gym.Wrapper.__init__(self, env)

        self.timestep = 0

    def step(self, action, **kwargs):
        self.timestep += 1
        obs, reward, done, info = self.env.step(action, **kwargs)
        if self.timestep >= self.max_timestep:
            done = True
            info["timeout"] = True
        else:
            info["timeout"] = False
        return obs, reward, done, info

    def reset(self, **kwargs):
        self.timestep = 0
        return self.env.reset(**kwargs)


class ObsTransformerWrapper(gym.Wrapper):
    def __init__(self, env):
        logger.info("[env type]:{}".format(type(env)))
        gym.Wrapper.__init__(self, env)

    def _get_obs(self, obs):
        # loads
        loads = []
        loads.append(obs.load_p)
        loads.append(obs.load_q)
        loads.append(obs.load_v)
        loads = np.concatenate(loads)

        # prods
        prods = []
        prods.append(obs.gen_p)
        prods.append(obs.gen_q)
        prods.append(obs.gen_v)
        prods = np.concatenate(prods)

        # rho
        rho = np.array(obs.rho) - 1.0

        next_load = obs.nextstep_load_p

        # action_space
        action_space_low = obs.action_space['adjust_gen_p'].low.tolist()
        action_space_high = obs.action_space['adjust_gen_p'].high.tolist()
        action_space_low[settings.balanced_id] = 0.0
        action_space_high[settings.balanced_id] = 0.0

        features = np.concatenate([
            loads, prods,
            rho.tolist(), next_load, action_space_low, action_space_high
        ])

        return features

    def step(self, action, **kwargs):
        self.raw_obs, reward, done, info = self.env.step(action, **kwargs)
        obs = self._get_obs(self.raw_obs)
        return obs, reward, done, info

    def reset(self, **kwargs):
        self.raw_obs = self.env.reset(**kwargs)
        obs = self._get_obs(self.raw_obs)
        return obs


class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env):
        logger.info("[env type]:{}".format(type(env)))
        gym.Wrapper.__init__(self, env)

    def step(self, action, **kwargs):
        obs, reward, done, info = self.env.step(action, **kwargs)

        shaping_reward = 1.0

        info["origin_reward"] = reward

        return obs, shaping_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ActionWrapper(gym.Wrapper):
    def __init__(self, env, raw_env):
        logger.info("[env type]:{}".format(type(env)))
        gym.Wrapper.__init__(self, env)
        self.raw_env = raw_env
        self.v_action = np.zeros(self.raw_env.settings.num_gen)

    def step(self, action, **kwargs):
        N = len(action)

        gen_p_action_space = self.env.raw_obs.action_space['adjust_gen_p']

        low_bound = gen_p_action_space.low
        high_bound = gen_p_action_space.high

        mapped_action = low_bound + (action - (-1.0)) * (
            (high_bound - low_bound) / 2.0)
        mapped_action[self.raw_env.settings.balanced_id] = 0.0
        mapped_action = np.clip(mapped_action, low_bound, high_bound)

        ret_action = form_action(mapped_action, self.v_action)
        return self.env.step(ret_action, **kwargs)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def get_env():
    env = Environment(settings, "EPRIReward")
    env.action_space = None
    raw_env = env

    env = MaxTimestepWrapper(env)
    env = RewardShapingWrapper(env)
    env = ObsTransformerWrapper(env)
    env = ActionWrapper(env, raw_env)

    return env
