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

import unittest
import threading
import time
import parl
import numpy as np
from parl.remote.master import Master
from parl.remote.worker import Worker
from parl.remote.client import disconnect
from parl.utils import logger
import gym
from gym.spaces import Box, Discrete


@parl.remote_class
class RemoteGymEnv(object):
    def __init__(self, env_name=None):
        assert isinstance(env_name, str)

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

        class ObservationSpace(object):
            def __init__(self, observation_space, low, high, shape=None):
                self.observation_space = observation_space
                self.low = low
                self.high = high
                self.shape = shape

        self.env = gym.make(env_name)
        self._max_episode_steps = int(self.env._max_episode_steps)
        self._elapsed_steps = int(self.env._elapsed_steps)

        self.observation_space = ObservationSpace(
            self.env.observation_space, self.env.observation_space.low,
            self.env.observation_space.high, self.env.observation_space.shape)
        if isinstance(self.env.action_space, Discrete):
            self.action_space = ActionSpace(n=self.env.action_space.n)
        elif isinstance(self.env.action_space, Box):
            self.action_space = ActionSpace(
                self.env.action_space, self.env.action_space.low,
                self.env.action_space.high, self.env.action_space.shape)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def seed(self, seed):
        return self.env.seed(seed)

    def render(self):
        return logger.warning('Using remote env, no need to render')


class TestRemoteEnv(unittest.TestCase):
    def tearDown(self):
        disconnect()

    def test_discrete_env_wrapper(self):
        logger.info("Running: test discrete_env_wrapper")
        master = Master(port=8267)
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(3)
        woker1 = Worker('localhost:8267', 1)

        parl.connect('localhost:8267')
        logger.info("Running: test discrete_env_wrapper: 1")

        env = RemoteGymEnv(env_name='MountainCar-v0')
        env.seed(1)
        env.render()

        obs, done = env.reset(), False
        observation_space = env.observation_space
        obs_space_high = observation_space.high
        obs_space_low = observation_space.low
        self.assertEqual(obs_space_high[0], 0.6)
        self.assertEqual(obs_space_low[0], -1.2)

        action_space = env.action_space
        act_dim = action_space.n
        self.assertEqual(act_dim, 3)

        # Run an episode with a random policy
        total_steps, episode_reward = 0, 0
        while not done:
            action = np.random.choice(act_dim)
            next_obs, reward, done, _ = env.step(action)
            episode_reward += reward
        logger.info('Episode done, total_steps {}, episode_reward {}'.format(
            total_steps, episode_reward))

        master.exit()
        woker1.exit()

    def test_continuous_env_wrapper(self):
        logger.info("Running: test continuous_env_wrapper")
        master = Master(port=8268)
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(3)
        woker1 = Worker('localhost:8268', 1)

        parl.connect('localhost:8268')
        logger.info("Running: test continuous_env_wrapper: 1")

        env = RemoteGymEnv(env_name='Pendulum-v0')
        env.seed(0)
        env.render()

        obs, done = env.reset(), False
        observation_space = env.observation_space
        obs_space_high = observation_space.high
        obs_space_low = observation_space.low
        self.assertEqual(obs_space_high[1], 1.)
        self.assertEqual(obs_space_low[1], -1.)

        action_space = env.action_space
        action_space_high = action_space.high
        action_space_low = action_space.low
        self.assertEqual(action_space_high, [2.])
        self.assertEqual(action_space_low, [-2.])

        # Run an episode with a random policy
        total_steps, episode_reward = 0, 0
        while not done:
            total_steps += 1
            action = env.action_space.sample()
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward
        logger.info('Episode done, total_steps {}, episode_reward {}'.format(
            total_steps, episode_reward))

        master.exit()
        woker1.exit()


if __name__ == '__main__':
    unittest.main()
