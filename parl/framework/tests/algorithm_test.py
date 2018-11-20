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

import paddle.fluid as fluid
import parl.layers as layers
from parl.framework.model_base import Model
from parl.framework.algorithm_base import Algorithm
from copy import deepcopy
import numpy as np
import unittest
import sys

class Value(Model):
    def __init__(self, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        self.fc1 = layers.fc(size=256, act='relu')
        self.fc2 = layers.fc(size=128, act='relu')
        self.fc3 = layers.fc(size=self.act_dim)
    
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
            outputs = executor.run(pred_program,
                    feed={'obs': x},
                    fetch_list=[dqn.q_value, dqn.q_target_value])
            self.assertNotEqual(outputs[0].flatten(), outputs[1].flatten())
        critic_model.sync_paras_to(dqn.target_model)

        random_obs = np.random.random(size=(N, 4)).astype('float32')
        for i in range(N):
            x = np.expand_dims(random_obs[i], axis=0)
            outputs = executor.run(pred_program,
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
            outputs = executor.run(pred_program,
                    feed={'obs': x},
                    fetch_list=[dqn.q_value])

            outputs_2 = executor.run(pred_program_2,
                    feed={'obs_2': x},
                    fetch_list=[dqn_2.q_value])
            self.assertNotEqual(outputs[0].flatten(), outputs_2[0].flatten())
        dqn.critic_model.sync_paras_to(dqn_2.critic_model)

        random_obs = np.random.random(size=(N, 4)).astype('float32')
        for i in range(N):
            x = np.expand_dims(random_obs[i], axis=0)
            outputs = executor.run(pred_program,
                    feed={'obs': x},
                    fetch_list=[dqn.q_value])

            outputs_2 = executor.run(pred_program_2,
                    feed={'obs_2': x},
                    fetch_list=[dqn_2.q_value])
            self.assertEqual(outputs[0].flatten(), outputs_2[0].flatten())

if __name__ == '__main__':
    unittest.main()
