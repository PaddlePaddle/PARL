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

import os
import numpy as np
import numpy.random as random
import paddle.fluid as fluid
from parl import layers
from parl import Agent
from parl.utils import get_gpu_count, machine_info


class ElevatorAgent(Agent):
    def __init__(self, algorithm, obs_dim, action_dim):
        self._action_dim = action_dim
        self._obs_dim = obs_dim
        self._update_target_steps = 1000

        self._global_step = 0
        self.exploration_ratio = 0.9
        self.exploration_decre = 1e-7
        self.exploration_min = 0.1
        super(ElevatorAgent, self).__init__(algorithm)

        use_cuda = machine_info.is_gpu_available()
        if self.gpu_id >= 0:
            assert get_gpu_count() == 1, 'Only support training in single GPU,\
                    Please set environment variable: `export CUDA_VISIBLE_DEVICES=[GPU_ID_YOU_WANT_TO_USE]` .'

        else:
            os.environ['CPU_NUM'] = str(1)

        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.num_threads = 1
        exec_strategy.num_iteration_per_drop_scope = 10
        build_strategy = fluid.BuildStrategy()
        build_strategy.remove_unnecessary_lock = False

        self.learn_pe = fluid.ParallelExecutor(
            use_cuda=use_cuda,
            main_program=self.learn_program,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy,
        )

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):
            obs = layers.data(
                name='obs', shape=[self._obs_dim], dtype='float32')
            self._value = self.alg.define_predict(obs)

        with fluid.program_guard(self.learn_program):
            obs = layers.data(
                name='obs', shape=[self._obs_dim], dtype='float32')
            action = layers.data(name='act', shape=[1], dtype='int32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs', shape=[self._obs_dim], dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            self._cost = self.alg.define_learn(obs, action, reward, next_obs,
                                               terminal)

    def sample(self, obs):
        if self.exploration_ratio > self.exploration_min:
            self.exploration_ratio -= self.exploration_decre
        q_values = self.predict(obs)

        ret_actions = list()
        for i in range(len(q_values)):  # number of elevators
            if (random.random() < self.exploration_ratio):
                action = random.randint(0, self._action_dim)
            else:
                action = np.argmax(q_values[i])
            ret_actions.append(int(action))
        return ret_actions

    def predict(self, obs):
        pred_Q = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self._value])
        return pred_Q[0]

    def learn(self, obs, act, reward, next_obs, terminal):
        self._global_step += 1
        if self._global_step % self._update_target_steps == 0:
            self.alg.sync_target(self.gpu_id)

        feed = {
            'obs': obs.astype('float32'),
            'act': act.astype('int32'),
            'reward': reward,
            'next_obs': next_obs.astype('float32'),
            'terminal': terminal
        }
        cost = self.learn_pe.run(feed=feed, fetch_list=[self._cost.name])[0]
        return cost
