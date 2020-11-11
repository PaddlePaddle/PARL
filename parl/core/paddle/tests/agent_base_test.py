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
import unittest
import paddle
import paddle.nn as nn
import parl
import os


class TestModel(parl.Model):
    def __init__(self):
        super(TestModel, self).__init__()
        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def predict(self, obs):
        out = self.fc1(obs)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


class ACModel(parl.Model):  # TODO: use a new file to test this model
    def __init__(self):
        super(ACModel, self).__init__()
        self.actor = Actor()
        self.critic = Critic()

    def predict(self, obs):
        return self.actor(obs)

    def Q(self, obs):
        return self.critic(obs)


class Actor(parl.Model):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(4, 300)
        self.fc2 = nn.Linear(300, 2)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out


class Critic(parl.Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(4, 300)
        self.fc2 = nn.Linear(300, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out


class TestAlgorithm(parl.Algorithm):
    def __init__(self, model):
        self.model = model

    def predict(self, obs):
        return self.model.predict(obs)

    def learn(self, obs, label):
        pred_output = self.model.policy(obs)
        mse_loss = paddle.nn.loss.MSELoss()
        cost = mse_loss(pred_output, label)
        cost = paddle.reduce_mean(cost)  # TODO: paddle.mean
        return cost


class TestAgent(parl.Agent):
    def __init__(self, algorithm):
        super(TestAgent, self).__init__(algorithm)

    def predict(self, obs):
        obs = paddle.to_tensor(obs.astype(np.float32))
        output = self.alg.predict(obs)
        output_np = output.numpy()
        return output_np

    def learn(self, obs, label):
        obs = paddle.to_tensor(obs.astype(np.float32))
        label = paddle.to_tensor(label.astype(np.float32))
        output = self.alg.learn(obs, label)
        output_np = output.numpy()
        return output_np


class AgentBaseTest(unittest.TestCase):
    def setUp(self):
        self.model = TestModel()
        self.alg = TestAlgorithm(self.model)

    def test_agent(self):
        agent = TestAgent(self.alg)
        obs = np.random.random([10, 4]).astype('float32')
        output_np = agent.predict(obs)
        self.assertIsNotNone(output_np)

    def test_save(self):
        agent = TestAgent(self.alg)
        save_path1 = 'mymodel.ckpt'
        save_path2 = os.path.join('my_model', 'model-2.ckpt')
        agent.save(save_path1)
        agent.save(save_path2)
        self.assertTrue(os.path.exists(save_path1))
        self.assertTrue(os.path.exists(save_path2))

    def test_save_with_model(self):
        agent = TestAgent(self.alg)
        save_path1 = 'mymodel.ckpt'
        save_path2 = os.path.join('my_model', 'model-2.ckpt')
        agent.save(save_path1, agent.alg.model)
        agent.save(save_path2, agent.alg.model)
        self.assertTrue(os.path.exists(save_path1))
        self.assertTrue(os.path.exists(save_path2))

    def test_restore(self):
        agent = TestAgent(self.alg)
        obs = np.random.random([10, 4]).astype('float32')
        save_path1 = 'model.ckpt'
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

    def test_restore_with_model(self):
        agent = TestAgent(self.alg)
        obs = np.random.random([10, 4]).astype('float32')
        save_path1 = 'model.ckpt'
        previous_output = agent.predict(obs)
        agent.save(save_path1, agent.alg.model)
        agent.restore(save_path1, agent.alg.model)
        current_output = agent.predict(obs)
        np.testing.assert_equal(current_output, previous_output)

        # a new agent instance
        another_agent = TestAgent(self.alg)
        another_agent.restore(save_path1, agent.alg.model)
        current_output = another_agent.predict(obs)
        np.testing.assert_equal(current_output, previous_output)

    # def test_compiled_restore(self):
    #     agent = TestAgent(self.alg)
    #     agent.learn_program = parl.compile(agent.learn_program)
    #     obs = np.random.random([3, 10]).astype('float32')
    #     previous_output = agent.predict(obs)
    #     save_path1 = 'model.ckpt'
    #     agent.save(save_path1)
    #     agent.restore(save_path1)

    #     # a new agent instance
    #     another_agent = TestAgent(self.alg)
    #     another_agent.learn_program = parl.compile(another_agent.learn_program)
    #     another_agent.restore(save_path1)
    #     current_output = another_agent.predict(obs)
    #     np.testing.assert_equal(current_output, previous_output)


class ACAgentBaseTest(unittest.TestCase):
    def setUp(self):
        self.model = ACModel()
        self.alg = TestAlgorithm(self.model)

    def test_agent(self):
        agent = TestAgent(self.alg)
        obs = np.random.random([10, 4]).astype('float32')
        output_np = agent.predict(obs)
        self.assertIsNotNone(output_np)

    def test_save(self):
        agent = TestAgent(self.alg)
        save_path1 = 'mymodel.ckpt'
        save_path2 = os.path.join('my_model', 'model-2.ckpt')
        agent.save(save_path1)
        agent.save(save_path2)
        self.assertTrue(os.path.exists(save_path1))
        self.assertTrue(os.path.exists(save_path2))

    def test_save_with_model(self):
        agent = TestAgent(self.alg)
        save_path1 = 'mymodel.ckpt'
        save_path2 = os.path.join('my_model', 'model-2.ckpt')
        agent.save(save_path1, agent.alg.model)
        agent.save(save_path2, agent.alg.model)
        self.assertTrue(os.path.exists(save_path1))
        self.assertTrue(os.path.exists(save_path2))

    def test_restore(self):
        agent = TestAgent(self.alg)
        obs = np.random.random([10, 4]).astype('float32')
        save_path1 = 'model.ckpt'
        previous_output = agent.predict(obs)
        previous_q_np = agent.alg.model.Q(paddle.to_tensor(obs)).numpy()

        agent.save(save_path1)
        agent.restore(save_path1)
        current_output = agent.predict(obs)
        current_q_np = agent.alg.model.Q(paddle.to_tensor(obs)).numpy()
        np.testing.assert_equal(current_output, previous_output)
        np.testing.assert_equal(current_q_np, previous_q_np)

        # a new agent instance
        another_agent = TestAgent(self.alg)
        another_agent.restore(save_path1)
        current_output = another_agent.predict(obs)
        np.testing.assert_equal(current_output, previous_output)

    def test_restore_with_model(self):
        agent = TestAgent(self.alg)
        obs = np.random.random([10, 4]).astype('float32')
        save_path1 = 'model.ckpt'
        previous_output = agent.predict(obs)
        previous_q_np = agent.alg.model.Q(paddle.to_tensor(obs)).numpy()
        agent.save(save_path1, agent.alg.model)
        agent.restore(save_path1, agent.alg.model)
        current_output = agent.predict(obs)
        current_q_np = agent.alg.model.Q(paddle.to_tensor(obs)).numpy()
        np.testing.assert_equal(current_output, previous_output)
        np.testing.assert_equal(current_q_np, previous_q_np)

        # a new agent instance
        another_agent = TestAgent(self.alg)
        another_agent.restore(save_path1, agent.alg.model)
        current_output = another_agent.predict(obs)
        current_q_np = agent.alg.model.Q(paddle.to_tensor(obs)).numpy()
        np.testing.assert_equal(current_output, previous_output)
        np.testing.assert_equal(current_q_np, previous_q_np)

    def test_restore_with_actor_model(self):
        agent = TestAgent(self.alg)
        obs = np.random.random([10, 4]).astype('float32')
        save_path1 = 'model.ckpt'
        previous_output = agent.predict(obs)
        previous_q_np = agent.alg.model.Q(paddle.to_tensor(obs)).numpy()
        agent.save(save_path1, agent.alg.model.actor)
        agent.restore(save_path1, agent.alg.model.actor)
        current_output = agent.predict(obs)
        current_q_np = agent.alg.model.Q(paddle.to_tensor(obs)).numpy()
        np.testing.assert_equal(current_output, previous_output)
        np.testing.assert_equal(current_q_np == previous_q_np)

        # a new agent instance
        another_agent = TestAgent(self.alg)
        another_agent.restore(save_path1, agent.alg.model.actor)
        current_output = another_agent.predict(obs)
        current_q_np = agent.alg.model.Q(paddle.to_tensor(obs)).numpy()
        np.testing.assert_equal(current_output, previous_output)
        np.testing.assert_equal(current_q_np, previous_q_np)

    # def test_compiled_restore(self):
    #     agent = TestAgent(self.alg)
    #     agent.learn_program = parl.compile(agent.learn_program)
    #     obs = np.random.random([3, 10]).astype('float32')
    #     previous_output = agent.predict(obs)
    #     save_path1 = 'model.ckpt'
    #     agent.save(save_path1)
    #     agent.restore(save_path1)

    #     # a new agent instance
    #     another_agent = TestAgent(self.alg)
    #     another_agent.learn_program = parl.compile(another_agent.learn_program)
    #     another_agent.restore(save_path1)
    #     current_output = another_agent.predict(obs)
    #     np.testing.assert_equal(current_output, previous_output)


if __name__ == '__main__':
    unittest.main()
