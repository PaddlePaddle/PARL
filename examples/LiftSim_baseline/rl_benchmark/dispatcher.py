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

import sys
import parl
import numpy as np
import numpy.random as random

from copy import deepcopy
from collections import deque

from rlschool import EPSILON, HUGE
from rl_benchmark.model import RLDispatcherModel
from rl_benchmark.agent import ElevatorAgent
from parl.algorithms import DQN
from parl.utils import ReplayMemory

MEMORY_SIZE = 1000000
BATCH_SIZE = 64


class RL_dispatcher():
    """
    An RL benchmark for elevator system
    """

    def __init__(self, env, max_episode):
        self.env = env

        self._obs_dim = env.observation_space
        self._act_dim = env.action_space
        self._global_step = 0
        self.max_episode = max_episode
        self._rpm = ReplayMemory(MEMORY_SIZE, self._obs_dim, 1)
        self._model = RLDispatcherModel(self._act_dim)
        hyperparas = {
            'action_dim': self._act_dim,
            'lr': 5.0e-4,
            'gamma': 0.998
        }

        self._algorithm = DQN(self._model, hyperparas)
        self._agent = ElevatorAgent(self._algorithm, self._obs_dim,
                                    self._act_dim)
        self._warm_up_size = 2000
        self._statistic_freq = 1000
        self._loss_queue = deque()

    def run_episode(self):
        self.env.reset()
        acc_reward = 0.0

        while self._global_step < self.max_episode:
            # self.env.render()
            state = self.env.state
            action = self._agent.sample(state)
            state_, reward, done, info = self.env.step(action)
            output_info = self.learn_step(state, action, reward)
            acc_reward += reward
            if (isinstance(output_info, dict) and len(output_info) > 0):
                self.env.log_notice("%s", output_info)
            if (self._global_step % 3600 == 0):
                self.env.log_notice(
                    "Accumulated Reward: %f, Mansion Status: %s", acc_reward,
                    self.env.statistics)
                acc_reward = 0.0

        self._agent.save('./model.ckpt')

    def learn_step(self, state, action, r):
        self._global_step += 1
        if (self._global_step > self._warm_up_size):
            for i in range(self.env.elevator_num):
                self._rpm.append(self._last_observation_array[i],
                                 self._last_action[i], self._last_reward,
                                 deepcopy(state[i]), False)
        self._last_observation_array = deepcopy(state)
        self._last_action = deepcopy(action)
        self._last_reward = r

        ret_dict = {}
        if self._rpm.size() > self._warm_up_size:
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = \
                self._rpm.sample_batch(BATCH_SIZE)
            cost = self._agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs, batch_terminal)
            self._loss_queue.appendleft(cost)
            if (len(self._loss_queue) > self._statistic_freq):
                self._loss_queue.pop()
            if (self._global_step % self._statistic_freq == 0):
                ret_dict["Temporal Difference Error(Average)"] = \
                    float(sum(self._loss_queue)) / float(len(self._loss_queue))

        return ret_dict
