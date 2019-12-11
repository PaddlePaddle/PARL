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

import parl
import numpy as np
from parl import layers
from parl.utils import machine_info
from paddle import fluid


class OpenSimAgent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(OpenSimAgent, self).__init__(algorithm)

        # Use ParallelExecutor to make program running faster
        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.use_experimental_executor = True
        exec_strategy.num_threads = 4
        build_strategy = fluid.BuildStrategy()
        build_strategy.remove_unnecessary_lock = True

        with fluid.scope_guard(fluid.global_scope().new_scope()):
            self.learn_pe = fluid.ParallelExecutor(
                use_cuda=machine_info.is_gpu_available(),
                main_program=self.learn_program,
                exec_strategy=exec_strategy,
                build_strategy=build_strategy)

        with fluid.scope_guard(fluid.global_scope().new_scope()):
            self.pred_pe = fluid.ParallelExecutor(
                use_cuda=machine_info.is_gpu_available(),
                main_program=self.pred_program,
                exec_strategy=exec_strategy,
                build_strategy=build_strategy)

        # Attention: In the beginning, sync target model totally.
        self.alg.sync_target(
            decay=0, share_vars_parallel_executor=self.learn_pe)

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.pred_act = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            act = layers.data(
                name='act', shape=[self.act_dim], dtype='float32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs', shape=[self.obs_dim], dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            _, self.critic_cost = self.alg.learn(obs, act, reward, next_obs,
                                                 terminal)

    def predict(self, obs):
        feed = {'obs': obs}
        act = self.pred_pe.run(feed=[feed], fetch_list=[self.pred_act.name])[0]
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        feed = {
            'obs': obs,
            'act': act,
            'reward': reward,
            'next_obs': next_obs,
            'terminal': terminal
        }
        critic_cost = self.learn_pe.run(
            feed=[feed], fetch_list=[self.critic_cost.name])[0]
        self.alg.sync_target(share_vars_parallel_executor=self.learn_pe)
        return critic_cost
