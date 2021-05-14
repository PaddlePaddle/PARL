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

import numpy as np
import parl
from parl import layers
from paddle import fluid
from parl.utils import ReplayMemory
from parl.utils import machine_info, get_gpu_count


class MAAgent(parl.Agent):
    def __init__(self,
                 algorithm,
                 agent_index=None,
                 obs_dim_n=None,
                 act_dim_n=None,
                 batch_size=None,
                 speedup=False):
        assert isinstance(agent_index, int)
        assert isinstance(obs_dim_n, list)
        assert isinstance(act_dim_n, list)
        assert isinstance(batch_size, int)
        assert isinstance(speedup, bool)
        self.agent_index = agent_index
        self.obs_dim_n = obs_dim_n
        self.act_dim_n = act_dim_n
        self.batch_size = batch_size
        self.speedup = speedup
        self.n = len(act_dim_n)

        self.memory_size = int(1e6)
        self.min_memory_size = batch_size * 25  # batch_size * args.max_episode_len
        self.rpm = ReplayMemory(
            max_size=self.memory_size,
            obs_dim=self.obs_dim_n[agent_index],
            act_dim=self.act_dim_n[agent_index])
        self.global_train_step = 0

        if machine_info.is_gpu_available():
            assert get_gpu_count() == 1, 'Only support training in single GPU,\
                    Please set environment variable: `export CUDA_VISIBLE_DEVICES=[GPU_ID_TO_USE]` .'

        super(MAAgent, self).__init__(algorithm)

        # Attention: In the beginning, sync target model totally.
        self.alg.sync_target(decay=0)

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()
        self.next_q_program = fluid.Program()
        self.next_a_program = fluid.Program()

        with fluid.program_guard(self.pred_program):
            obs = layers.data(
                name='obs',
                shape=[self.obs_dim_n[self.agent_index]],
                dtype='float32')
            self.pred_act = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):
            obs_n = [
                layers.data(
                    name='obs' + str(i),
                    shape=[self.obs_dim_n[i]],
                    dtype='float32') for i in range(self.n)
            ]
            act_n = [
                layers.data(
                    name='act' + str(i),
                    shape=[self.act_dim_n[i]],
                    dtype='float32') for i in range(self.n)
            ]
            target_q = layers.data(name='target_q', shape=[], dtype='float32')
            self.critic_cost = self.alg.learn(obs_n, act_n, target_q)

        with fluid.program_guard(self.next_q_program):
            obs_n = [
                layers.data(
                    name='obs' + str(i),
                    shape=[self.obs_dim_n[i]],
                    dtype='float32') for i in range(self.n)
            ]
            act_n = [
                layers.data(
                    name='act' + str(i),
                    shape=[self.act_dim_n[i]],
                    dtype='float32') for i in range(self.n)
            ]
            self.next_Q = self.alg.Q_next(obs_n, act_n)

        with fluid.program_guard(self.next_a_program):
            obs = layers.data(
                name='obs',
                shape=[self.obs_dim_n[self.agent_index]],
                dtype='float32')
            self.next_action = self.alg.predict_next(obs)

        if self.speedup:
            self.pred_program = parl.compile(self.pred_program)
            self.learn_program = parl.compile(self.learn_program,
                                              self.critic_cost)
            self.next_q_program = parl.compile(self.next_q_program)
            self.next_a_program = parl.compile(self.next_a_program)

    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0)
        obs = obs.astype('float32')
        act = self.fluid_executor.run(
            self.pred_program, feed={'obs': obs},
            fetch_list=[self.pred_act])[0]
        return act[0]

    def learn(self, agents):
        self.global_train_step += 1

        # only update parameter every 100 steps
        if self.global_train_step % 100 != 0:
            return 0.0

        if self.rpm.size() <= self.min_memory_size:
            return 0.0

        batch_obs_n = []
        batch_act_n = []
        batch_obs_new_n = []

        rpm_sample_index = self.rpm.make_index(self.batch_size)
        for i in range(self.n):
            batch_obs, batch_act, _, batch_obs_new, _ \
                = agents[i].rpm.sample_batch_by_index(rpm_sample_index)
            batch_obs_n.append(batch_obs)
            batch_act_n.append(batch_act)
            batch_obs_new_n.append(batch_obs_new)
        _, _, batch_rew, _, batch_isOver \
                = self.rpm.sample_batch_by_index(rpm_sample_index)

        # compute target q
        target_q = 0.0
        target_act_next_n = []
        for i in range(self.n):
            feed = {'obs': batch_obs_new_n[i]}
            target_act_next = agents[i].fluid_executor.run(
                agents[i].next_a_program,
                feed=feed,
                fetch_list=[agents[i].next_action])[0]
            target_act_next_n.append(target_act_next)
        feed_obs = {'obs' + str(i): batch_obs_new_n[i] for i in range(self.n)}
        feed_act = {
            'act' + str(i): target_act_next_n[i]
            for i in range(self.n)
        }
        feed = feed_obs.copy()
        feed.update(feed_act)  # merge two dict
        target_q_next = self.fluid_executor.run(
            self.next_q_program, feed=feed, fetch_list=[self.next_Q])[0]
        target_q += (
            batch_rew + self.alg.gamma * (1.0 - batch_isOver) * target_q_next)

        feed_obs = {'obs' + str(i): batch_obs_n[i] for i in range(self.n)}
        feed_act = {'act' + str(i): batch_act_n[i] for i in range(self.n)}
        target_q = target_q.astype('float32')
        feed = feed_obs.copy()
        feed.update(feed_act)
        feed['target_q'] = target_q
        critic_cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.critic_cost])[0]

        self.alg.sync_target()
        return critic_cost

    def add_experience(self, obs, act, reward, next_obs, terminal):
        self.rpm.append(obs, act, reward, next_obs, terminal)
