#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


class ACModel(parl.Model):
    def __init__(self):
        super(ACModel, self).__init__()
        self.actor = Actor()
        self.critic = Critic()

    def predict(self, obs):
        return self.actor(obs)

    def Q(self, obs):
        return self.critic(obs)


class DoubleInputACModel(parl.Model):
    def __init__(self):
        super(DoubleInputACModel, self).__init__()
        self.actor = DoubleInputActor()
        self.critic = Critic()

    def predict(self, obs):
        return self.actor(obs)

    def Q(self, obs):
        return self.critic(obs)


class ACModelWithDropout(parl.Model):
    def __init__(self):
        super(ACModelWithDropout, self).__init__()
        self.actor = ActorWithDropout()
        self.critic = Critic()

    def predict(self, obs):
        return self.actor(obs)

    def Q(self, obs):
        return self.critic(obs)


class ACModelWithBN(parl.Model):
    def __init__(self):
        super(ACModelWithBN, self).__init__()
        self.actor = Actor()
        self.critic = CriticWithBN()

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


class ActorWithDropout(parl.Model):
    def __init__(self):
        super(ActorWithDropout, self).__init__()
        self.fc1 = nn.Linear(4, 300)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(300, 2)

    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout(out)
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


class CriticWithBN(parl.Model):
    def __init__(self):
        super(CriticWithBN, self).__init__()
        self.fc1 = nn.Linear(4, 300)
        self.bn = nn.BatchNorm1d(300)
        self.fc2 = nn.Linear(300, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn(out)
        out = self.fc2(out)
        return out


class DoubleInputActor(parl.Model):
    def __init__(self):
        super(DoubleInputActor, self).__init__()
        self.fc1 = nn.Linear(4, 300)
        self.fc2 = nn.Linear(300 + 4, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, x1, x2):
        out = self.fc1(x1)
        out = self.fc2(torch.concat([out, x2], 1))
        out = self.fc3(out)
        return out


class TestAlgorithm(parl.Algorithm):
    def __init__(self, model):
        super(TestAlgorithm, self).__init__(model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def predict(self, obs):
        return self.model.predict(obs)

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
        cost = self.alg.learn(obs, label)

    def predict(self, obs):
        return self.alg.predict(obs)


class ACAgentBaseTest(unittest.TestCase):
    def setUp(self):
        self.model = ACModel()
        self.alg = TestAlgorithm(self.model)
        self.target_model = ACModel()
        self.target_alg = TestAlgorithm(self.target_model)
        self.double_model = DoubleInputACModel()
        self.double_alg = TestAlgorithm(self.double_model)
        self.dropout_model = ACModelWithDropout()
        self.dropout_alg = TestAlgorithm(self.dropout_model)
        self.bn_model = ACModelWithBN()
        self.bn_alg = TestAlgorithm(self.bn_model)

    def test_agent(self):
        agent = TestAgent(self.alg)
        obs = torch.randn(10, 4)
        act_output = agent.predict(obs)
        self.assertIsNotNone(act_output)

    def test_save(self):
        agent = TestAgent(self.alg)
        save_path1 = 'model.ckpt'
        save_path2 = os.path.join('my_model', 'model-2.ckpt')
        agent.save(save_path1)
        agent.save(save_path2)
        self.assertTrue(os.path.exists(save_path1))
        self.assertTrue(os.path.exists(save_path2))

    def test_restore(self):
        agent = TestAgent(self.alg)
        obs = torch.randn(10, 4)
        output = agent.predict(obs)
        save_path1 = 'model.ckpt'
        previous_output = agent.predict(obs).detach().cpu().numpy()
        agent.save(save_path1)
        agent.restore(save_path1)
        current_output = agent.predict(obs).detach().cpu().numpy()
        np.testing.assert_equal(current_output, previous_output)

    def test_get_weights(self):
        agent = TestAgent(self.alg)
        weight = agent.get_weights()
        agent.set_weights(weight)

    def test_train_and_eval_mode(self):
        agent = TestAgent(self.alg)
        obs = torch.randn(10, 4)
        agent.train()
        self.assertTrue(agent.training)
        act_train = agent.predict(obs)
        q_train = agent.alg.model.Q(obs).detach().numpy()
        agent.eval()
        self.assertFalse(agent.training)
        act_eval = agent.predict(obs)
        q_eval = agent.alg.model.Q(obs).detach().numpy()
        self.assertTrue((act_train == act_eval).all())
        self.assertTrue((q_train == q_eval).all())

    def test_train_and_eval_mode_with_dropout(self):
        agent = TestAgent(self.dropout_alg)
        obs = torch.randn(10, 4)
        agent.train()
        self.assertTrue(agent.training)
        act_train = agent.predict(obs)
        q_train = agent.alg.model.Q(obs).detach().numpy()
        agent.eval()
        self.assertFalse(agent.training)
        act_eval = agent.predict(obs)
        q_eval = agent.alg.model.Q(obs).detach().numpy()
        self.assertFalse((act_train == act_eval).all())
        self.assertTrue((q_train == q_eval).all())

    def test_train_and_eval_mode_with_bn(self):
        agent = TestAgent(self.bn_alg)
        obs = torch.randn(10, 4)
        agent.train()
        self.assertTrue(agent.training)
        act_train = agent.predict(obs)
        q_train = agent.alg.model.Q(obs).detach().numpy()
        agent.eval()
        self.assertFalse(agent.training)
        act_eval = agent.predict(obs)
        q_eval = agent.alg.model.Q(obs).detach().numpy()
        self.assertTrue((act_train == act_eval).all())
        self.assertFalse((q_train == q_eval).all())


if __name__ == '__main__':
    unittest.main()
