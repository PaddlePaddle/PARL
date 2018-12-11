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
import paddle.fluid as fluid
import parl.layers as layers
from parl.framework.agent_base import Agent


class CartpoleAgent(Agent):
    def __init__(self, algorithm, obs_dim, act_dim, seed=1):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.seed = seed
        super(CartpoleAgent, self).__init__(algorithm)

    def build_program(self):
        self.pred_program = fluid.Program()
        self.train_program = fluid.Program()

        fluid.default_startup_program().random_seed = self.seed
        self.train_program.random_seed = self.seed

        with fluid.program_guard(self.pred_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.act_prob = self.alg.define_predict(obs)

        with fluid.program_guard(self.train_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            act = layers.data(name='act', shape=[1], dtype='int64')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            self.cost = self.alg.define_learn(obs, act, reward)

    def sample(self, obs):
        obs = np.expand_dims(obs, axis=0)
        act_prob = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.act_prob])[0]
        act_prob = np.squeeze(act_prob, axis=0)
        act = np.random.choice(range(self.act_dim), p=act_prob)
        return act

    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0)
        act_prob = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.act_prob])[0]
        act_prob = np.squeeze(act_prob, axis=0)
        act = np.argmax(act_prob)
        return act

    def learn(self, obs, act, reward):
        act = np.expand_dims(act, axis=-1)
        feed = {
            'obs': obs.astype('float32'),
            'act': act.astype('int64'),
            'reward': reward.astype('float32')
        }
        cost = self.fluid_executor.run(
            self.train_program, feed=feed, fetch_list=[self.cost])[0]
        return cost
