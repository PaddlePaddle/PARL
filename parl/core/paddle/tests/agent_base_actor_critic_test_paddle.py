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


class DoubleInputActor(parl.Model):
    def __init__(self):
        super(DoubleInputActor, self).__init__()
        self.fc1 = nn.Linear(4, 300)
        self.fc2 = nn.Linear(300 + 4, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, x1, x2):
        out = self.fc1(x1)
        out = self.fc2(paddle.concat([out, x2], 1))
        out = self.fc3(out)
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
        cost = paddle.mean(cost)
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


class ACAgentBaseTest(unittest.TestCase):
    def setUp(self):
        self.model = ACModel()
        self.alg = TestAlgorithm(self.model)
        self.target_model = ACModel()
        self.target_alg = TestAlgorithm(self.target_model)
        self.double_model = DoubleInputACModel()
        self.double_alg = TestAlgorithm(self.double_model)

    def test_agent(self):
        agent = TestAgent(self.alg)
        obs = np.random.random([10, 4]).astype('float32')
        act_np = agent.predict(obs)
        self.assertIsNotNone(act_np)

        params = agent.get_weights()
        hid = np.dot(obs,
                     params['actor.fc1.weight']) + params['actor.fc1.bias']
        act = np.dot(hid,
                     params['actor.fc2.weight']) + params['actor.fc2.bias']
        self.assertLess((act.sum() - act_np.sum()), 1e-5)

    def test_save(self):
        agent = TestAgent(self.alg)
        save_path1 = 'my_acmodel.ckpt'
        save_path2 = os.path.join('my_ac_model', 'model-2.ckpt')
        agent.save(save_path1)
        agent.save(save_path2)
        self.assertTrue(os.path.exists(save_path1))
        self.assertTrue(os.path.exists(save_path2))

    def test_inference_model(self):
        agent = TestAgent(self.alg)
        save_path1 = 'my_acmodel'
        save_path2 = os.path.join('my_ac_model', 'model-2')
        input_spec = [
            paddle.static.InputSpec(shape=[None, 4], dtype='float32')
        ]
        input_shape = [[None, 4]]
        input_type = ['float32']
        agent.save_inference_model(save_path1, input_shape, input_type,
                                   self.model.actor)
        agent.save_inference_model(save_path2, input_shape, input_type,
                                   self.model.actor)
        self.assertTrue(os.path.exists(save_path1 + '.pdmodel'))
        self.assertTrue(os.path.exists(save_path2 + '.pdmodel'))

    def test_double_inference_model(self):
        agent = TestAgent(self.double_alg)
        save_path1 = 'my_double_model'
        save_path2 = os.path.join('my_double_model', 'model-2')
        input_shape = [[None, 4], [None, 4]]
        input_type = ['float32', 'float32']
        agent.save_inference_model(save_path1, input_shape, input_type,
                                   self.double_model.actor)
        agent.save_inference_model(save_path2, input_shape, input_type,
                                   self.double_model.actor)
        self.assertTrue(os.path.exists(save_path1 + '.pdmodel'))
        self.assertTrue(os.path.exists(save_path2 + '.pdmodel'))

    def test_save_with_model(self):
        agent = TestAgent(self.alg)
        save_path1 = 'my_acmodel.ckpt'
        save_path2 = os.path.join('my_ac_model', 'model-2.ckpt')
        agent.save(save_path1, agent.alg.model)
        agent.save(save_path2, agent.alg.model)
        self.assertTrue(os.path.exists(save_path1))
        self.assertTrue(os.path.exists(save_path2))

    def test_restore(self):
        agent = TestAgent(self.alg)
        obs = np.random.random([10, 4]).astype('float32')
        save_path1 = 'my_acmodel.ckpt'
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
        save_path1 = 'my_acmodel.ckpt'
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
        save_path1 = 'my_acmodel.ckpt'
        previous_output = agent.predict(obs)
        previous_q_np = agent.alg.model.Q(paddle.to_tensor(obs)).numpy()
        agent.save(save_path1, agent.alg.model.actor)
        agent.restore(save_path1, agent.alg.model.actor)
        current_output = agent.predict(obs)
        current_q_np = agent.alg.model.Q(paddle.to_tensor(obs)).numpy()
        np.testing.assert_equal(current_output, previous_output)
        np.testing.assert_equal(current_q_np, previous_q_np)

        # a new agent instance
        another_agent = TestAgent(self.alg)
        another_agent.restore(save_path1, agent.alg.model.actor)
        current_output = another_agent.predict(obs)
        current_q_np = agent.alg.model.Q(paddle.to_tensor(obs)).numpy()
        np.testing.assert_equal(current_output, previous_output)
        np.testing.assert_equal(current_q_np, previous_q_np)

    def test_get_weights(self):
        agent = TestAgent(self.alg)
        params = agent.get_weights()
        expected_params = list(agent.alg.model.named_parameters())
        self.assertEqual(len(params), len(expected_params))
        for i, key in enumerate(params):
            self.assertLess(
                (params[key].sum() - expected_params[i][1].numpy().sum()),
                1e-5)

    def test_set_weights(self):
        agent = TestAgent(self.alg)
        target_agent = TestAgent(self.target_alg)

        params = agent.get_weights()
        for key in params.keys():
            params[key] = params[key] + 1.0

        target_agent.set_weights(params)

        for i, j in zip(params.values(), target_agent.get_weights().values()):
            self.assertLessEqual(abs(i.sum() - j.sum()), 1e-3)

    def test_set_weights_between_different_models(self):
        agent = TestAgent(self.alg)
        target_agent = TestAgent(self.target_alg)

        N = 10
        obs = np.random.randn(N, 4)
        out = agent.predict(obs)
        target_out = target_agent.predict(obs)
        self.assertNotEqual(out.sum(), target_out.sum())

        params = agent.get_weights()
        target_agent.set_weights(params)

        obs = np.random.randn(N, 4)
        out = agent.predict(obs)
        target_out = target_agent.predict(obs)
        self.assertEqual(out.sum(), target_out.sum())

    def test_set_weights_with_wrong_params_num(self):
        agent = TestAgent(self.alg)
        params = agent.get_weights()
        del params['actor.fc2.bias']
        del params['critic.fc2.bias']
        with self.assertRaises(AssertionError):
            agent.set_weights(params)

    def test_set_weights_with_wrong_params_shape(self):
        agent = TestAgent(self.alg)
        params = agent.get_weights()
        params['actor.fc1.weight'] = params['actor.fc2.bias']
        params['critic.fc1.weight'] = params['critic.fc2.bias']
        with self.assertRaises(AssertionError):
            agent.set_weights(params)

    def test_set_weights_with_modified_params(self):
        agent = TestAgent(self.alg)
        params = agent.get_weights()
        params['actor.fc1.weight'][0][0] = 100
        params['actor.fc1.bias'][0] = 100
        params['actor.fc2.weight'][0][0] = 100
        params['actor.fc2.bias'][0] = 100
        params['critic.fc1.weight'][0][0] = 100
        params['critic.fc1.bias'][0] = 100
        params['critic.fc2.weight'][0][0] = 100
        params['critic.fc2.bias'][0] = 100
        agent.set_weights(params)
        new_params = agent.get_weights()
        for i, j in zip(params.values(), new_params.values()):
            self.assertLessEqual(abs(i.sum() - j.sum()), 1e-3)


if __name__ == '__main__':
    unittest.main()
