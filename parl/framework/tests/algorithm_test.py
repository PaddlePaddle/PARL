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
import sys
import unittest
from copy import deepcopy
from paddle.fluid import ParamAttr
from paddle.fluid.executor import global_scope
from parl.framework.algorithm_base import Algorithm
from parl.framework.model_base import Model


class Value(Model):
    def __init__(self, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.fc1 = layers.fc(
            size=256,
            act=None,
            param_attr=ParamAttr(name='fc1.w'),
            bias_attr=ParamAttr(name='fc1.b'))
        self.fc2 = layers.fc(
            size=128,
            act=None,
            param_attr=ParamAttr(name='fc2.w'),
            bias_attr=ParamAttr(name='fc2.b'))
        self.fc3 = layers.fc(
            size=self.act_dim,
            act=None,
            param_attr=ParamAttr(name='fc3.w'),
            bias_attr=ParamAttr(name='fc3.b'))

    def value(self, obs):
        out = self.fc1(obs)
        out = self.fc2(out)
        value = self.fc3(out)
        return value


class QLearning(Algorithm):
    def __init__(self, critic_model):
        self.critic_model = critic_model
        self.target_model = deepcopy(critic_model)

    def define_predict(self, obs):
        self.q_value = self.critic_model.value(obs)
        self.q_target_value = self.target_model.value(obs)


class AlgorithmBaseTest(unittest.TestCase):
    def test_sync_paras_in_one_program(self):
        critic_model = Value(obs_dim=4, act_dim=1)
        dqn = QLearning(critic_model)
        pred_program = fluid.Program()
        with fluid.program_guard(pred_program):
            obs = layers.data(name='obs', shape=[4], dtype='float32')
            dqn.define_predict(obs)
        place = fluid.CUDAPlace(0)
        executor = fluid.Executor(place)
        executor.run(fluid.default_startup_program())

        N = 10
        random_obs = np.random.random(size=(N, 4)).astype('float32')
        for i in range(N):
            x = np.expand_dims(random_obs[i], axis=0)
            outputs = executor.run(
                pred_program,
                feed={'obs': x},
                fetch_list=[dqn.q_value, dqn.q_target_value])
            self.assertNotEqual(outputs[0].flatten(), outputs[1].flatten())
        critic_model.sync_paras_to(dqn.target_model)

        random_obs = np.random.random(size=(N, 4)).astype('float32')
        for i in range(N):
            x = np.expand_dims(random_obs[i], axis=0)
            outputs = executor.run(
                pred_program,
                feed={'obs': x},
                fetch_list=[dqn.q_value, dqn.q_target_value])
            self.assertEqual(outputs[0].flatten(), outputs[1].flatten())

    def test_sync_paras_among_programs(self):
        critic_model = Value(obs_dim=4, act_dim=1)
        dqn = QLearning(critic_model)
        dqn_2 = deepcopy(dqn)
        pred_program = fluid.Program()
        pred_program_2 = fluid.Program()
        with fluid.program_guard(pred_program):
            obs = layers.data(name='obs', shape=[4], dtype='float32')
            dqn.define_predict(obs)

        # algorithm #2
        with fluid.program_guard(pred_program_2):
            obs_2 = layers.data(name='obs_2', shape=[4], dtype='float32')
            dqn_2.define_predict(obs_2)

        place = fluid.CUDAPlace(0)
        executor = fluid.Executor(place)
        executor.run(fluid.default_startup_program())

        N = 10
        random_obs = np.random.random(size=(N, 4)).astype('float32')
        for i in range(N):
            x = np.expand_dims(random_obs[i], axis=0)
            outputs = executor.run(
                pred_program, feed={'obs': x}, fetch_list=[dqn.q_value])

            outputs_2 = executor.run(
                pred_program_2, feed={'obs_2': x}, fetch_list=[dqn_2.q_value])
            self.assertNotEqual(outputs[0].flatten(), outputs_2[0].flatten())
        dqn.critic_model.sync_paras_to(dqn_2.critic_model)

        random_obs = np.random.random(size=(N, 4)).astype('float32')
        for i in range(N):
            x = np.expand_dims(random_obs[i], axis=0)
            outputs = executor.run(
                pred_program, feed={'obs': x}, fetch_list=[dqn.q_value])

            outputs_2 = executor.run(
                pred_program_2, feed={'obs_2': x}, fetch_list=[dqn_2.q_value])
            self.assertEqual(outputs[0].flatten(), outputs_2[0].flatten())

    def test_sync_paras_with_decay(self):
        critic_model = Value(obs_dim=4, act_dim=1)
        dqn = QLearning(critic_model)
        pred_program = fluid.Program()
        with fluid.program_guard(pred_program):
            obs = layers.data(name='obs', shape=[4], dtype='float32')
            dqn.define_predict(obs)
        place = fluid.CUDAPlace(0)
        executor = fluid.Executor(place)
        executor.run(fluid.default_startup_program())

        model_fc1_w = np.array(global_scope().find_var('fc1.w').get_tensor())
        model_fc1_b = np.array(global_scope().find_var('fc1.b').get_tensor())
        model_fc2_w = np.array(global_scope().find_var('fc2.w').get_tensor())
        model_fc2_b = np.array(global_scope().find_var('fc2.b').get_tensor())
        model_fc3_w = np.array(global_scope().find_var('fc3.w').get_tensor())
        model_fc3_b = np.array(global_scope().find_var('fc3.b').get_tensor())

        unique_id = dqn.target_model.parameter_names[0].split('_')[-1]
        target_model_fc1_w = np.array(global_scope().find_var(
            'PARL_target_fc1.w_{}'.format(unique_id)).get_tensor())
        target_model_fc1_b = np.array(global_scope().find_var(
            'PARL_target_fc1.b_{}'.format(unique_id)).get_tensor())
        target_model_fc2_w = np.array(global_scope().find_var(
            'PARL_target_fc2.w_{}'.format(unique_id)).get_tensor())
        target_model_fc2_b = np.array(global_scope().find_var(
            'PARL_target_fc2.b_{}'.format(unique_id)).get_tensor())
        target_model_fc3_w = np.array(global_scope().find_var(
            'PARL_target_fc3.w_{}'.format(unique_id)).get_tensor())
        target_model_fc3_b = np.array(global_scope().find_var(
            'PARL_target_fc3.b_{}'.format(unique_id)).get_tensor())

        decay = 0.9
        critic_model.sync_paras_to(dqn.target_model, decay=decay)

        # update target_model parameters value in numpy way
        target_model_fc1_w = decay * target_model_fc1_w + (
            1 - decay) * model_fc1_w
        target_model_fc1_b = decay * target_model_fc1_b + (
            1 - decay) * model_fc1_b
        target_model_fc2_w = decay * target_model_fc2_w + (
            1 - decay) * model_fc2_w
        target_model_fc2_b = decay * target_model_fc2_b + (
            1 - decay) * model_fc2_b
        target_model_fc3_w = decay * target_model_fc3_w + (
            1 - decay) * model_fc3_w
        target_model_fc3_b = decay * target_model_fc3_b + (
            1 - decay) * model_fc3_b

        N = 10
        random_obs = np.random.random(size=(N, 4)).astype('float32')
        for i in range(N):
            x = np.expand_dims(random_obs[i], axis=0)
            real_target_outputs = executor.run(
                pred_program, feed={'obs': x},
                fetch_list=[dqn.q_target_value])[0]

            # Ideal target output
            out_np = np.dot(x, target_model_fc1_w) + target_model_fc1_b
            out_np = np.dot(out_np, target_model_fc2_w) + target_model_fc2_b
            out_np = np.dot(out_np, target_model_fc3_w) + target_model_fc3_b

            self.assertLess(float(np.abs(real_target_outputs - out_np)), 1e-5)


if __name__ == '__main__':
    unittest.main()
