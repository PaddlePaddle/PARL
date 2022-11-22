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

# Simplified version of https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/component/envs.py
# Mujoco wrapper for single agent

import numpy as np
import gym
import time
from parl.env.compat_wrappers import CompatWrapper

__all__ = ['wrap_rms']


class TimeLimitMaskEnv(gym.Wrapper):
    """ Env wrapper that marks bad_transition
    """

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True
        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class MonitorEnv(gym.Wrapper):
    """ Env wrapper that keeps tracks of total raw episode rewards, length of raw episode rewards for evaluation.
    """

    def __init__(self, env):
        super().__init__(env)
        self.tstart = time.time()
        self.rewards = None

    def step(self, action):
        ob, rew, done, info = self.env.step(action)
        self.update(ob, rew, done, info)
        return ob, rew, done, info

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


class RunningMeanStd(object):
    """ Calculating running mean and variance
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """

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


class VecNormalizeEnv(gym.Wrapper):
    """ Env wrapper that normalize reward, observation based on running mean return and running mean variance.
    """

    def __init__(self,
                 env,
                 ob=True,
                 ret=True,
                 clipob=10.,
                 cliprew=10.,
                 gamma=0.99,
                 epsilon=1e-8):
        super().__init__(env)
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

        ob = self._obfilt(ob)
        # normalize reward
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rew = np.clip(rew / np.sqrt(self.ret_rms.var + self.epsilon),
                          -self.cliprew, self.cliprew)

        if new:
            self.ret = np.zeros(1)

        return ob, rew, new, info

    def reset(self):
        self.ret = np.zeros(1)
        ob = self.env.reset()
        return self._obfilt(ob)

    def _obfilt(self, ob, update=True):
        if self.ob_rms:
            # normalize observation, expand batch dim if observation does not have batch dimension
            if ob.ndim == 1:
                ob = np.expand_dims(ob, 0)
            if self.training and update:
                self.ob_rms.update(ob)
            ob = np.clip((ob - self.ob_rms.mean) /
                         np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob,
                         self.clipob)
            # squeeze observation shape after normalization
            if ob.shape[0] == 1:
                ob = np.squeeze(ob, axis=0)
            return ob
        else:
            return ob

    def get_ob_rms(self):
        return self.ob_rms

    def set_ob_rms(self, ob_rms):
        self.ob_rms = ob_rms

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


def get_wrapper_by_cls(venv, cls):
    """ Fetch env wrapper class cls from current venv
    Args:
        venv (gym.Wrapper): current env
        cls (gym.Wrapper): target env wrapper class
    """
    if isinstance(venv, cls):
        return venv
    elif hasattr(venv, 'env'):
        return get_wrapper_by_cls(venv.env, cls)

    return None


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var,
                                       batch_count):
    """ helper function that updates batch mean, variance, count
    Args:
        mean (np.array): current mean
        var (np.array): current variance
        count (float): current count
        batch_mean (np.array): batch mean
        batch_var (np.array): batch variance
        batch_count (int): batch size
    """
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


def get_ob_rms(env):
    """ get ob_rms value from current env, if current env does not wrap VecNormalizeEnv, None will be returned
    Args:
        env (gym.Wrapper): current env
    """
    vec_norm_env = get_wrapper_by_cls(env, VecNormalizeEnv)
    ob_rms = None
    if vec_norm_env:
        ob_rms = vec_norm_env.get_ob_rms()

    return ob_rms


def wrap_rms(env, gamma, test=False, ob_rms=None):
    """ Wrap original Mujoco environment with wrapper envs to provide normalization using rms and extra functionality,
    rewards information are stored in info['episode']. This is the wrapper for single agent.
    Args:
        env (gym.Wrapper): Mujoco env
        gamma (float or None): discounting factor, if test then gamma = None
        test (bool): True if test else False
        ob_rms (None or np.array): ob_rms from training environment, not None only when test is True
    """
    env = CompatWrapper(env)
    env = TimeLimitMaskEnv(env)
    env = MonitorEnv(env)

    if test:
        env = VecNormalizeEnv(env, ret=False)
        env.eval()
        env.set_ob_rms(ob_rms)
    else:
        env = VecNormalizeEnv(env, gamma=gamma)

    return env
