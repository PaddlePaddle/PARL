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


class AtariAgent(parl.Agent):
    def __init__(self,
                 algorithm,
                 obs_shape,
                 predict_thread_num,
                 learn_data_provider=None):
        """

        Args:
            algorithm (`parl.Algorithm`): a2c algorithm
            obs_shape (list/tuple): observation shape of atari environment
            predict_thread_num (int): number of predict thread (predict parallel exector)
            learn_data_provider: data generator of training
        """

        assert isinstance(obs_shape, (list, tuple))
        assert isinstance(predict_thread_num, int)
        self.obs_shape = obs_shape

        super(AtariAgent, self).__init__(algorithm)

        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.use_experimental_executor = True
        exec_strategy.num_threads = 4
        build_strategy = fluid.BuildStrategy()
        build_strategy.remove_unnecessary_lock = True

        # Use ParallelExecutor to make learn program run faster
        self.learn_exe = fluid.ParallelExecutor(
            use_cuda=machine_info.is_gpu_available(),
            loss_name=self.learn_outputs[0],
            main_program=self.learn_program,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)

        self.sample_exes = []
        for _ in range(predict_thread_num):
            with fluid.scope_guard(fluid.global_scope().new_scope()):
                pe = fluid.ParallelExecutor(
                    use_cuda=machine_info.is_gpu_available(),
                    main_program=self.sample_program,
                    build_strategy=build_strategy,
                    exec_strategy=exec_strategy)
                self.sample_exes.append(pe)

        if learn_data_provider:
            self.learn_reader.decorate_tensor_provider(learn_data_provider)
            self.learn_reader.start()

    def build_program(self):
        self.sample_program = fluid.Program()
        self.predict_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.sample_program):
            obs = layers.data(
                name='obs', shape=self.obs_shape, dtype='float32')
            sample_actions, values = self.alg.sample(obs)
            self.sample_outputs = [sample_actions.name, values.name]

        with fluid.program_guard(self.predict_program):
            obs = layers.data(
                name='obs', shape=self.obs_shape, dtype='float32')
            self.predict_actions = self.alg.predict(obs)

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
                name='entropy_coeff', shape=[], dtype='float32')

            self.learn_reader = fluid.layers.create_py_reader_by_data(
                capacity=32,
                feed_list=[
                    obs, actions, advantages, target_values, lr, entropy_coeff
                ])
            obs, actions, advantages, target_values, lr, entropy_coeff = fluid.layers.read_file(
                self.learn_reader)

            total_loss, pi_loss, vf_loss, entropy = self.alg.learn(
                obs, actions, advantages, target_values, lr, entropy_coeff)
            self.learn_outputs = [
                total_loss.name, pi_loss.name, vf_loss.name, entropy.name
            ]

    def sample(self, obs_np, thread_id):
        """
        Args:
            obs_np: a numpy float32 array of shape ([B] + observation_space)
                    Format of image input should be NCHW format.

        Returns:
            sample_ids: a numpy int64 array of shape [B]
            values: a numpy float32 array of shape [B]
        """
        obs_np = obs_np.astype('float32')

        sample_actions, values = self.sample_exes[thread_id].run(
            feed={'obs': obs_np}, fetch_list=self.sample_outputs)
        return sample_actions, values

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
        total_loss, pi_loss, vf_loss, entropy = self.learn_exe.run(
            fetch_list=self.learn_outputs)
        return total_loss, pi_loss, vf_loss, entropy
