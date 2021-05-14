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
import parl
from parl import layers
from parl.utils import machine_info
from parl.utils.scheduler import PiecewiseScheduler, LinearDecayScheduler


class AtariAgent(parl.Agent):
    def __init__(self, algorithm, config):
        """

        Args:
            algorithm (`parl.Algorithm`): algorithm to be used in this agent.
            config (dict): config file describing the training hyper-parameters(see a2c_config.py)
        """

        self.obs_shape = config['obs_shape']
        super(AtariAgent, self).__init__(algorithm)

        self.lr_scheduler = LinearDecayScheduler(config['start_lr'],
                                                 config['max_sample_steps'])

        self.entropy_coeff_scheduler = PiecewiseScheduler(
            config['entropy_coeff_scheduler'])

    def build_program(self):
        self.sample_program = fluid.Program()
        self.predict_program = fluid.Program()
        self.value_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.sample_program):
            obs = layers.data(
                name='obs', shape=self.obs_shape, dtype='float32')
            sample_actions, values = self.alg.sample(obs)
            self.sample_outputs = [sample_actions, values]

        with fluid.program_guard(self.predict_program):
            obs = layers.data(
                name='obs', shape=self.obs_shape, dtype='float32')
            self.predict_actions = self.alg.predict(obs)

        with fluid.program_guard(self.value_program):
            obs = layers.data(
                name='obs', shape=self.obs_shape, dtype='float32')
            self.values = self.alg.value(obs)

        with fluid.program_guard(self.learn_program):
            obs = layers.data(
                name='obs', shape=self.obs_shape, dtype='float32')
            actions = layers.data(name='actions', shape=[], dtype='int64')
            advantages = layers.data(
                name='advantages', shape=[], dtype='float32')
            target_values = layers.data(
                name='target_values', shape=[], dtype='float32')
            lr = layers.data(
                name='lr', shape=[1], dtype='float32', append_batch_size=False)
            entropy_coeff = layers.data(
                name='entropy_coeff',
                shape=[1],
                dtype='float32',
                append_batch_size=False)

            total_loss, pi_loss, vf_loss, entropy = self.alg.learn(
                obs, actions, advantages, target_values, lr, entropy_coeff)
            self.learn_outputs = [total_loss, pi_loss, vf_loss, entropy]
        self.learn_program = parl.compile(self.learn_program, total_loss)

    def sample(self, obs_np):
        """
        Args:
            obs_np: a numpy float32 array of shape ([B] + observation_space).
                    Format of image input should be NCHW format.

        Returns:
            sample_ids: a numpy int64 array of shape [B]
            values: a numpy float32 array of shape [B]
        """
        obs_np = obs_np.astype('float32')

        sample_actions, values = self.fluid_executor.run(
            self.sample_program,
            feed={'obs': obs_np},
            fetch_list=self.sample_outputs)
        return sample_actions, values

    def predict(self, obs_np):
        """
        Args:
            obs_np: a numpy float32 array of shape ([B] + observation_space).
                    Format of image input should be NCHW format.

        Returns:
            sample_ids: a numpy int64 array of shape [B]
        """
        obs_np = obs_np.astype('float32')

        predict_actions = self.fluid_executor.run(
            self.predict_program,
            feed={'obs': obs_np},
            fetch_list=[self.predict_actions])[0]
        return predict_actions

    def value(self, obs_np):
        """
        Args:
            obs_np: a numpy float32 array of shape ([B] + observation_space).
                    Format of image input should be NCHW format.

        Returns:
            values: a numpy float32 array of shape [B]
        """
        obs_np = obs_np.astype('float32')

        values = self.fluid_executor.run(
            self.value_program, feed={'obs': obs_np},
            fetch_list=[self.values])[0]
        return values

    def learn(self, obs_np, actions_np, advantages_np, target_values_np):
        """
        Args:
            obs_np: a numpy float32 array of shape ([B] + observation_space).
                    Format of image input should be NCHW format.
            actions_np: a numpy int64 array of shape [B]
            advantages_np: a numpy float32 array of shape [B]
            target_values_np: a numpy float32 array of shape [B]
        """

        obs_np = obs_np.astype('float32')
        actions_np = actions_np.astype('int64')
        advantages_np = advantages_np.astype('float32')
        target_values_np = target_values_np.astype('float32')

        lr = self.lr_scheduler.step(step_num=obs_np.shape[0])
        entropy_coeff = self.entropy_coeff_scheduler.step()

        total_loss, pi_loss, vf_loss, entropy = self.fluid_executor.run(
            self.learn_program,
            feed={
                'obs': obs_np,
                'actions': actions_np,
                'advantages': advantages_np,
                'target_values': target_values_np,
                'lr': np.array([lr], dtype='float32'),
                'entropy_coeff': np.array([entropy_coeff], dtype='float32')
            },
            fetch_list=self.learn_outputs)
        return total_loss, pi_loss, vf_loss, entropy, lr, entropy_coeff
