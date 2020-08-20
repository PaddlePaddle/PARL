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

# Simplified version of https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/component/envs.py

import numpy as np
import gym
from gym.core import Wrapper
import time


class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True
        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class MonitorEnv(gym.Wrapper):
    def __init__(self, env):
        Wrapper.__init__(self, env=env)
        self.tstart = time.time()
        self.rewards = None

    def step(self, action):
        ob, rew, done, info = self.env.step(action)
        self.update(ob, rew, done, info)
        return (ob, rew, done, info)

    def update(self, ob, rew, done, info):
        self.rewards.append(rew)
        if done:
            eprew = sum(self.rewards)
            eplen = len(self.rewards)
            epinfo = {
                "r": round(eprew, 6),
                "l": eplen,
                "t": round(time.time() - self.tstart, 6)
            }
            assert isinstance(info, dict)
            info['episode'] = epinfo
            self.reset()

    def reset(self, **kwargs):
        self.rewards = []
        return self.env.reset(**kwargs)


class VectorEnv(gym.Wrapper):
    def step(self, action):
        ob, rew, done, info = self.env.step(action)
        ob = np.array(ob)
        ob = ob[np.newaxis, :]
        rew = np.array([rew])

        done = np.array([done])

        info = [info]
        return (ob, rew, done, info)


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var,
            batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var,
                                       batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class VecNormalize(gym.Wrapper):
    def __init__(self,
                 env,
                 ob=True,
                 ret=True,
                 clipob=10.,
                 cliprew=10.,
                 gamma=0.99,
                 epsilon=1e-8):
        Wrapper.__init__(self, env=env)
        observation_space = env.observation_space.shape[0]

        self.ob_rms = RunningMeanStd(shape=observation_space) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None

        self.clipob = clipob
        self.cliprew = cliprew
        self.gamma = gamma
        self.epsilon = epsilon
        self.ret = np.zeros(1)
        self.training = True

    def step(self, action):
        ob, rew, new, info = self.env.step(action)
        self.ret = self.ret * self.gamma + rew
        # normalize observation
        ob = self._obfilt(ob)
        # normalize reward
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rew = np.clip(rew / np.sqrt(self.ret_rms.var + self.epsilon),
                          -self.cliprew, self.cliprew)
        self.ret[new] = 0.
        return ob, rew, new, info

    def reset(self):
        self.ret = np.zeros(1)
        ob = self.env.reset()
        return self._obfilt(ob)

    def _obfilt(self, ob, update=True):
        if self.ob_rms:
            if self.training and update:
                self.ob_rms.update(ob)
            ob = np.clip((ob - self.ob_rms.mean) /
                         np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob,
                         self.clipob)
            return ob
        else:
            return ob

    def train(self):
        self.training = True

    def eval(self):
        self.trainint = False


def make_env(env_name, seed, gamma):
    env = gym.make(env_name)
    env.seed(seed)
    env = TimeLimitMask(env)
    env = MonitorEnv(env)
    env = VectorEnv(env)
    if gamma is None:
        env = VecNormalize(env, ret=False)
    else:
        env = VecNormalize(env, gamma=gamma)

    return env
