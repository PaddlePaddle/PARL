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
from parl.utils import logger


class MujocoAgent(parl.Agent):
    def __init__(self,
                 algorithm,
                 obs_dim,
                 act_dim,
                 kl_targ,
                 loss_type,
                 beta=1.0,
                 epsilon=0.2,
                 policy_learn_times=20,
                 value_learn_times=10,
                 value_batch_size=256):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        assert loss_type == 'CLIP' or loss_type == 'KLPEN'
        self.loss_type = loss_type
        super(MujocoAgent, self).__init__(algorithm)

        self.policy_learn_times = policy_learn_times
        # Adaptive kl penalty coefficient
        self.beta = beta
        self.kl_targ = kl_targ

        self.value_learn_times = value_learn_times
        self.value_batch_size = value_batch_size
        self.value_learn_buffer = None

    def build_program(self):
        self.policy_predict_program = fluid.Program()
        self.policy_sample_program = fluid.Program()
        self.policy_learn_program = fluid.Program()
        self.value_predict_program = fluid.Program()
        self.value_learn_program = fluid.Program()

        with fluid.program_guard(self.policy_sample_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            sampled_act = self.alg.sample(obs)
            self.policy_sample_output = [sampled_act]

        with fluid.program_guard(self.policy_predict_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            means = self.alg.predict(obs)
            self.policy_predict_output = [means]

        with fluid.program_guard(self.policy_learn_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            actions = layers.data(
                name='actions', shape=[self.act_dim], dtype='float32')
            advantages = layers.data(
                name='advantages', shape=[1], dtype='float32')
            if self.loss_type == 'KLPEN':
                beta = layers.data(name='beta', shape=[], dtype='float32')
                loss, kl = self.alg.policy_learn(obs, actions, advantages,
                                                 beta)
            else:
                loss, kl = self.alg.policy_learn(obs, actions, advantages)

            self.policy_learn_output = [loss, kl]

        with fluid.program_guard(self.value_predict_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            value = self.alg.value_predict(obs)
            self.value_predict_output = [value]

        with fluid.program_guard(self.value_learn_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            val = layers.data(name='val', shape=[], dtype='float32')
            value_loss = self.alg.value_learn(obs, val)
            self.value_learn_output = [value_loss]

    def policy_sample(self, obs):
        feed = {'obs': obs}
        sampled_act = self.fluid_executor.run(
            self.policy_sample_program,
            feed=feed,
            fetch_list=self.policy_sample_output)[0]
        return sampled_act

    def policy_predict(self, obs):
        feed = {'obs': obs}
        means = self.fluid_executor.run(
            self.policy_predict_program,
            feed=feed,
            fetch_list=self.policy_predict_output)[0]
        return means

    def value_predict(self, obs):
        feed = {'obs': obs}
        value = self.fluid_executor.run(
            self.value_predict_program,
            feed=feed,
            fetch_list=self.value_predict_output)[0]
        return value

    def _batch_policy_learn(self, obs, actions, advantages):
        if self.loss_type == 'KLPEN':
            feed = {
                'obs': obs,
                'actions': actions,
                'advantages': advantages,
                'beta': self.beta
            }
        else:
            feed = {'obs': obs, 'actions': actions, 'advantages': advantages}
        [loss, kl] = self.fluid_executor.run(
            self.policy_learn_program,
            feed=feed,
            fetch_list=self.policy_learn_output)
        return loss, kl

    def _batch_value_learn(self, obs, val):
        feed = {'obs': obs, 'val': val}
        value_loss = self.fluid_executor.run(
            self.value_learn_program,
            feed=feed,
            fetch_list=self.value_learn_output)[0]
        return value_loss

    def policy_learn(self, obs, actions, advantages):
        """ Learn policy:

        1. Sync parameters of policy model to old policy model
        2. Fix old policy model, and learn policy model multi times
        3. if use KLPEN loss, Adjust kl loss coefficient: beta
        """
        self.alg.sync_old_policy()

        all_loss, all_kl = [], []
        for _ in range(self.policy_learn_times):
            loss, kl = self._batch_policy_learn(obs, actions, advantages)
            all_loss.append(loss)
            all_kl.append(kl)

        if self.loss_type == 'KLPEN':
            # Adative KL penalty coefficient
            if kl > self.kl_targ * 2:
                self.beta = 1.5 * self.beta
            elif kl < self.kl_targ / 2:
                self.beta = self.beta / 1.5

        return np.mean(all_loss), np.mean(all_kl)

    def value_learn(self, obs, value):
        """ Fit model to current data batch + previous data batch
        """
        data_size = obs.shape[0]

        if self.value_learn_buffer is None:
            obs_train, value_train = obs, value
        else:
            obs_train = np.concatenate([obs, self.value_learn_buffer[0]])
            value_train = np.concatenate([value, self.value_learn_buffer[1]])
        self.value_learn_buffer = (obs, value)

        all_loss = []
        for _ in range(self.value_learn_times):
            random_ids = np.arange(obs_train.shape[0])
            np.random.shuffle(random_ids)
            shuffle_obs_train = obs_train[random_ids]
            shuffle_value_train = value_train[random_ids]
            start = 0
            while start < data_size:
                end = start + self.value_batch_size
                value_loss = self._batch_value_learn(
                    shuffle_obs_train[start:end, :],
                    shuffle_value_train[start:end])
                all_loss.append(value_loss)
                start += self.value_batch_size
        return np.mean(all_loss)
