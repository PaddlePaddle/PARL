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

import numpy as np
import paddle.fluid as fluid
import parl
from parl import layers

IMAGE_SIZE = (84, 84)
CONTEXT_LEN = 4


class AtariAgent(parl.Agent):
    def __init__(self, algorithm, act_dim, update_freq):
        super(AtariAgent, self).__init__(algorithm)
        assert isinstance(act_dim, int)
        self.act_dim = act_dim
        self.exploration = 1.0
        self.global_step = 0
        self.update_target_steps = 10000 // 4
        self.update_freq = update_freq

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):
            obs = layers.data(
                name='obs',
                shape=[CONTEXT_LEN, IMAGE_SIZE[0], IMAGE_SIZE[1]],
                dtype='float32')
            self.value = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):
            obs = layers.data(
                name='obs',
                shape=[CONTEXT_LEN, IMAGE_SIZE[0], IMAGE_SIZE[1]],
                dtype='float32')
            action = layers.data(name='act', shape=[1], dtype='int32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs',
                shape=[CONTEXT_LEN, IMAGE_SIZE[0], IMAGE_SIZE[1]],
                dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            sample_weight = layers.data(
                name='sample_weight', shape=[1], dtype='float32')
            self.cost, self.delta = self.alg.learn(
                obs, action, reward, next_obs, terminal, sample_weight)

    def sample(self, obs, decay_exploration=True):
        sample = np.random.random()
        if sample < self.exploration:
            act = np.random.randint(self.act_dim)
        else:
            if np.random.random() < 0.01:
                act = np.random.randint(self.act_dim)
            else:
                obs = np.expand_dims(obs, axis=0)
                pred_Q = self.fluid_executor.run(
                    self.pred_program,
                    feed={'obs': obs.astype('float32')},
                    fetch_list=[self.value])[0]
                pred_Q = np.squeeze(pred_Q, axis=0)
                act = np.argmax(pred_Q)
        if decay_exploration:
            self.exploration = max(0.1, self.exploration - 1e-6)
        return act

    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0)
        pred_Q = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.value])[0]
        pred_Q = np.squeeze(pred_Q, axis=0)
        act = np.argmax(pred_Q)
        return act

    def learn(self, obs, act, reward, next_obs, terminal, sample_weight):
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1

        act = np.expand_dims(act, -1)
        reward = np.clip(reward, -1, 1)
        feed = {
            'obs': obs.astype('float32'),
            'act': act.astype('int32'),
            'reward': reward.astype('float32'),
            'next_obs': next_obs.astype('float32'),
            'terminal': terminal.astype('bool'),
            'sample_weight': sample_weight.astype('float32')
        }
        cost, delta = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.cost, self.delta])
        return cost, delta
