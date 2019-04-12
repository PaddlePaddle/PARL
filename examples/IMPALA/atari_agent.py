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


class AtariAgent(Agent):
    def __init__(self, algorithm, config, learn_data_provider=None):
        self.config = config
        super(AtariAgent, self).__init__(algorithm)

        use_cuda = True if self.gpu_id >= 0 else False

        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.use_experimental_executor = True
        exec_strategy.num_threads = 4
        build_strategy = fluid.BuildStrategy()
        build_strategy.remove_unnecessary_lock = True

        # Use ParallelExecutor to make learn program running faster
        self.learn_exe = fluid.ParallelExecutor(
            use_cuda=use_cuda,
            main_program=self.learn_program,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)

        if learn_data_provider:
            self.learn_reader.decorate_tensor_provider(learn_data_provider)
            self.learn_reader.start()

    def build_program(self):
        self.sample_program = fluid.Program()
        self.predict_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.sample_program):
            obs = layers.data(
                name='obs', shape=self.config['obs_shape'], dtype='float32')
            self.sample_actions, self.behaviour_logits = self.alg.sample(obs)

        with fluid.program_guard(self.predict_program):
            obs = layers.data(
                name='obs', shape=self.config['obs_shape'], dtype='float32')
            self.predict_actions = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):
            obs = layers.data(
                name='obs', shape=self.config['obs_shape'], dtype='float32')
            actions = layers.data(name='actions', shape=[], dtype='int64')
            behaviour_logits = layers.data(
                name='behaviour_logits',
                shape=[self.config['act_dim']],
                dtype='float32')
            rewards = layers.data(name='rewards', shape=[], dtype='float32')
            dones = layers.data(name='dones', shape=[], dtype='float32')
            lr = layers.data(
                name='lr', shape=[1], dtype='float32', append_batch_size=False)
            entropy_coeff = layers.data(
                name='entropy_coeff', shape=[], dtype='float32')

            self.learn_reader = fluid.layers.create_py_reader_by_data(
                capacity=self.config['train_batch_size'],
                feed_list=[
                    obs, actions, behaviour_logits, rewards, dones, lr,
                    entropy_coeff
                ])

            obs, actions, behaviour_logits, rewards, dones, lr, entropy_coeff = fluid.layers.read_file(
                self.learn_reader)

            vtrace_loss, kl = self.alg.learn(obs, actions, behaviour_logits,
                                             rewards, dones, lr, entropy_coeff)
            self.learn_outputs = [
                vtrace_loss.total_loss.name, vtrace_loss.pi_loss.name,
                vtrace_loss.vf_loss.name, vtrace_loss.entropy.name, kl.name
            ]

    def sample(self, obs_np):
        """
        Args:
            obs_np: a numpy float32 array of shape ([B] + observation_space).
            Format of image input should be NCHW format.

        Returns:
            sample_ids: a numpy int64 array of shape [B]
        """
        obs_np = obs_np.astype('float32')

        sample_actions, behaviour_logits = self.fluid_executor.run(
            self.sample_program,
            feed={'obs': obs_np},
            fetch_list=[self.sample_actions, self.behaviour_logits])
        return sample_actions, behaviour_logits

    def predict(self, obs_np):
        """
        Args:
            obs_np: a numpy float32 array of shape ([B] + observation_space)
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

    def learn(self):
        total_loss, pi_loss, vf_loss, entropy, kl = self.learn_exe.run(
            fetch_list=self.learn_outputs)
        return total_loss, pi_loss, vf_loss, entropy, kl
