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
import unittest
from paddle import fluid
from parl import layers
import parl
import os


class TestModel(parl.Model):
    def __init__(self):
        self.fc1 = layers.fc(size=256)
        self.fc2 = layers.fc(size=1)

    def policy(self, obs):
        out = self.fc1(obs)
        out = self.fc2(out)
        return out


class TestAlgorithm(parl.Algorithm):
    def __init__(self, model):
        self.model = model

    def predict(self, obs):
        return self.model.policy(obs)

    def learn(self, obs, label):
        pred_output = self.model.policy(obs)
        cost = layers.square_error_cost(obs, label)
        cost = fluid.layers.reduce_mean(cost)
        return cost


class TestAgent(parl.Agent):
    def __init__(self, algorithm):
        super(TestAgent, self).__init__(algorithm)

    def build_program(self):
        self.predict_program = fluid.Program()
        self.learn_program = fluid.Program()
        with fluid.program_guard(self.predict_program):
            obs = layers.data(name='obs', shape=[10], dtype='float32')
            output = self.alg.predict(obs)
        self.predict_output = [output]

        with fluid.program_guard(self.learn_program):
            obs = layers.data(name='obs', shape=[10], dtype='float32')
            label = layers.data(name='label', shape=[1], dtype='float32')
            cost = self.alg.learn(obs, label)

    def learn(self, obs, label):
        output_np = self.fluid_executor.run(
            self.learn_program, feed={
                'obs': obs,
                'label': label
            })

    def predict(self, obs):
        output_np = self.fluid_executor.run(
            self.predict_program,
            feed={'obs': obs},
            fetch_list=self.predict_output)[0]
        return output_np


class AgentBaseTest(unittest.TestCase):
    def setUp(self):
        self.model = TestModel()
        self.alg = TestAlgorithm(self.model)

    def test_agent(self):
        agent = TestAgent(self.alg)
        obs = np.random.random([3, 10]).astype('float32')
        output_np = agent.predict(obs)
        self.assertIsNotNone(output_np)

    def test_save(self):
        agent = TestAgent(self.alg)
        save_path1 = 'model_dir'
        save_path2 = os.path.join('my_model', 'model-2_dir')
        agent.save(save_path1)
        agent.save(save_path2)
        self.assertTrue(os.path.exists(save_path1 + '/predict_program'))
        self.assertTrue(os.path.exists(save_path1 + '/learn_program'))
        self.assertTrue(os.path.exists(save_path2 + '/predict_program'))
        self.assertTrue(os.path.exists(save_path2 + '/learn_program'))

    def test_save_with_program(self):
        agent = TestAgent(self.alg)
        save_program_path1 = 'program_dir1'
        save_program_path2 = 'program_dir2'
        save_program_deep_path1 = os.path.join('my_program1', 'model-2_dir')
        save_program_deep_path2 = os.path.join('my_program2', 'model-2_dir')
        agent.save(save_program_path1, program=agent.predict_program)
        agent.save(save_program_path2, program=agent.learn_program)
        agent.save(save_program_deep_path1, program=agent.predict_program)
        agent.save(save_program_deep_path2, program=agent.learn_program)
        self.assertTrue(
            os.path.exists(save_program_path1 + '/predict_program'))
        self.assertTrue(os.path.exists(save_program_path2 + '/learn_program'))
        self.assertTrue(
            os.path.exists(save_program_deep_path1 + '/predict_program'))
        self.assertTrue(
            os.path.exists(save_program_deep_path2 + '/learn_program'))

    def test_restore(self):
        agent = TestAgent(self.alg)
        obs = np.random.random([3, 10]).astype('float32')

        save_path1 = 'model_dir'
        previous_output = agent.predict(obs)
        agent.save(save_path1)
        agent.restore(save_path1)
        current_output = agent.predict(obs)
        np.testing.assert_equal(current_output, previous_output)

        # a new agent instance
        another_agent = TestAgent(self.alg)
        another_agent.restore(save_path1)
        current_output = another_agent.predict(obs)
        np.testing.assert_equal(current_output, previous_output)

    def test_restore_with_program(self):
        agent = TestAgent(self.alg)
        obs = np.random.random([3, 10]).astype('float32')

        save_program_path1 = 'program_dir'
        previous_output = agent.predict(obs)
        agent.save(save_program_path1, program=agent.learn_program)
        agent.restore(save_program_path1, program=agent.learn_program)
        current_output = agent.predict(obs)
        np.testing.assert_equal(current_output, previous_output)

        # a new agent instance
        another_agent = TestAgent(self.alg)
        another_agent.restore(
            save_program_path1, program=another_agent.learn_program)
        current_output = another_agent.predict(obs)
        np.testing.assert_equal(current_output, previous_output)

    def test_compiled_restore(self):
        agent = TestAgent(self.alg)
        agent.learn_program = parl.compile(agent.learn_program)
        obs = np.random.random([3, 10]).astype('float32')

        save_path1 = 'model_dir'
        previous_output = agent.predict(obs)
        agent.save(save_path1)
        agent.restore(save_path1)

        # a new agent instance
        another_agent = TestAgent(self.alg)
        another_agent.learn_program = parl.compile(another_agent.learn_program)
        another_agent.restore(save_path1)
        current_output = another_agent.predict(obs)
        np.testing.assert_equal(current_output, previous_output)

    def test_compiled_restore_with_program(self):
        agent = TestAgent(self.alg)
        agent.learn_program = parl.compile(agent.learn_program)
        obs = np.random.random([3, 10]).astype('float32')

        save_program_path1 = 'program_dir'
        previous_output = agent.predict(obs)
        agent.save(save_program_path1, program=agent.learn_program)
        agent.restore(save_program_path1, program=agent.learn_program)

        # a new agent instance
        another_agent = TestAgent(self.alg)
        another_agent.learn_program = parl.compile(another_agent.learn_program)
        another_agent.restore(
            save_program_path1, program=another_agent.learn_program)
        current_output = another_agent.predict(obs)
        np.testing.assert_equal(current_output, previous_output)


if __name__ == '__main__':
    unittest.main()
