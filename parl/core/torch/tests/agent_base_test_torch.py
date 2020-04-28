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
import os

import torch
import torch.nn as nn
import torch.optim as optim

import parl


class TestModel(parl.Model):
    def __init__(self):
        super(TestModel, self).__init__()
        self.fc1 = nn.Linear(10, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, obs):
        out = self.fc1(obs)
        out = self.fc2(out)
        return out


class TestAlgorithm(parl.Algorithm):
    def __init__(self, model):
        super(TestAlgorithm, self).__init__(model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def predict(self, obs):
        return self.model(obs)

    def learn(self, obs, label):
        pred_output = self.model(obs)
        cost = (pred_output - obs).pow(2)
        self.optimizer.zero_grad()
        cost.backward()
        self.optimizer.step()
        return cost.item()


class TestAgent(parl.Agent):
    def __init__(self, algorithm):
        super(TestAgent, self).__init__(algorithm)

    def learn(self, obs, label):
        cost = self.algorithm.learn(obs, label)

    def predict(self, obs):
        return self.algorithm.predict(obs)


class AgentBaseTest(unittest.TestCase):
    def setUp(self):
        self.model = TestModel()
        self.alg = TestAlgorithm(self.model)

    def test_agent(self):
        agent = TestAgent(self.alg)
        obs = torch.randn(3, 10)
        output = agent.predict(obs)
        self.assertIsNotNone(output)

    def test_save(self):
        agent = TestAgent(self.alg)
        obs = torch.randn(3, 10)
        save_path1 = 'model.ckpt'
        save_path2 = os.path.join('my_model', 'model-2.ckpt')
        agent.save(save_path1)
        agent.save(save_path2)
        self.assertTrue(os.path.exists(save_path1))
        self.assertTrue(os.path.exists(save_path2))

    def test_restore(self):
        agent = TestAgent(self.alg)
        obs = torch.randn(3, 10)
        output = agent.predict(obs)
        save_path1 = 'model.ckpt'
        previous_output = agent.predict(obs).detach().cpu().numpy()
        agent.save(save_path1)
        agent.restore(save_path1)
        current_output = agent.predict(obs).detach().cpu().numpy()
        np.testing.assert_equal(current_output, previous_output)

    def test_weights(self):
        agent = TestAgent(self.alg)
        weight = agent.get_weights()
        agent.set_weights(weight)


if __name__ == '__main__':
    unittest.main()
