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

    def forward(self, obs):
        out = self.fc1(obs)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


class TestModelWithoutForward(parl.Model):
    def __init__(self):
        super(TestModelWithoutForward, self).__init__()
        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.forward = None

    def predict(self, obs):
        out = self.fc1(obs)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


class TestModelWithDropout(parl.Model):
    def __init__(self):
        super(TestModelWithDropout, self).__init__()
        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, obs):
        out = self.fc1(obs)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out


class TestModelWithBN(parl.Model):
    def __init__(self):
        super(TestModelWithBN, self).__init__()
        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.bn = nn.BatchNorm1D(128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, obs):
        out = self.fc1(obs)
        out = self.fc2(out)
        out = self.bn(out)
        out = self.fc3(out)
        return out


class DoubleInputTestModel(parl.Model):
    def __init__(self):
        super(DoubleInputTestModel, self).__init__()
        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256 + 4, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, obs_1, obs_2):
        out = self.fc1(obs_1)
        out = self.fc2(paddle.concat([out, obs_2], 1))
        out = self.fc3(out)
        return out


class TestAlgorithm(parl.Algorithm):
    def __init__(self, model):
        self.model = model

    def predict(self, obs):
        return self.model(obs)

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


class AgentBaseTest(unittest.TestCase):
    def setUp(self):
        self.model = TestModel()
        self.alg = TestAlgorithm(self.model)
        self.target_model = TestModel()
        self.target_alg = TestAlgorithm(self.target_model)
        self.target_model_without_forward = TestModelWithoutForward()
        self.target_alg_without_forward = TestAlgorithm(
            self.target_model_without_forward)
        self.double_model = DoubleInputTestModel()
        self.double_alg = TestAlgorithm(self.double_model)

    def test_agent(self):
        agent = TestAgent(self.alg)
        obs = np.random.random([10, 4]).astype('float32')
        output_np = agent.predict(obs)
        self.assertIsNotNone(output_np)

        params = agent.get_weights()
        hd1 = np.dot(obs, params['fc1.weight']) + params['fc1.bias']
        hd2 = np.dot(hd1, params['fc2.weight']) + params['fc2.bias']
        out = np.dot(hd2, params['fc3.weight']) + params['fc3.bias']
        self.assertLess((out.sum() - output_np.sum()), 1e-5)

    def test_save(self):
        agent = TestAgent(self.alg)
        save_path1 = 'mymodel.ckpt'
        save_path2 = os.path.join('my_model', 'model-2.ckpt')
        agent.save(save_path1)
        agent.save(save_path2)
        self.assertTrue(os.path.exists(save_path1))
        self.assertTrue(os.path.exists(save_path2))

    def test_save_inference_model(self):
        agent = TestAgent(self.alg)
        save_path1 = 'my_inference_model'
        save_path2 = os.path.join('my_infer_model', 'model-2')
        input_shapes = [[None, 4]]
        input_dtypes = ['float32']
        agent.save_inference_model(save_path1, input_shapes, input_dtypes)
        agent.save_inference_model(save_path2, input_shapes, input_dtypes)
        self.assertTrue(os.path.exists(save_path1 + '.pdmodel'))
        self.assertTrue(os.path.exists(save_path2 + '.pdmodel'))
        agent_without_forward = TestAgent(self.target_alg_without_forward)
        input_shapes = [[None, 4]]
        input_dtypes = ['float32']
        with self.assertRaises(AssertionError):
            agent_without_forward.save_inference_model(
                save_path1, input_shapes, input_dtypes)
        input_shapes = (None, 4)
        input_dtypes = ['float32']
        with self.assertRaises(AssertionError):
            agent.save_inference_model(save_path1, input_shapes, input_dtypes)
        input_shapes = [[None, 4]]
        input_dtypes = 'float32'
        with self.assertRaises(AssertionError):
            agent.save_inference_model(save_path1, input_shapes, input_dtypes)
        input_shapes = [[None, 4]]
        input_dtypes = ['float32', 'float32']
        with self.assertRaises(AssertionError):
            agent.save_inference_model(save_path1, input_shapes, input_dtypes)

    def test_save_inference_model_with_multi_inputs(self):
        agent = TestAgent(self.double_alg)
        save_path1 = 'my_inference_model_with_multi_inputs'
        save_path2 = os.path.join('my_infer_model_with_multi_inputs',
                                  'model-2')
        input_shapes = [[None, 4], [None, 4]]
        input_dtypes = ['float32', 'float32']
        agent.save_inference_model(save_path1, input_shapes, input_dtypes)
        agent.save_inference_model(save_path2, input_shapes, input_dtypes)
        self.assertTrue(os.path.exists(save_path1 + '.pdmodel'))
        self.assertTrue(os.path.exists(save_path2 + '.pdmodel'))
        input_shapes = (None, 4)
        input_dtypes = ['float32', 'float32']
        with self.assertRaises(AssertionError):
            agent.save_inference_model(save_path1, input_shapes, input_dtypes)
        input_shapes = [[None, 4]]
        input_dtypes = 'float32'
        with self.assertRaises(AssertionError):
            agent.save_inference_model(save_path1, input_shapes, input_dtypes)
        input_shapes = [[None, 4]]
        input_dtypes = ['float32', 'float32']
        with self.assertRaises(AssertionError):
            agent.save_inference_model(save_path1, input_shapes, input_dtypes)

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
        save_path1 = 'mymodel.ckpt'
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
        save_path1 = 'mymodel.ckpt'
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
        del params['fc2.bias']
        with self.assertRaises(AssertionError):
            agent.set_weights(params)

    def test_set_weights_with_wrong_params_shape(self):
        agent = TestAgent(self.alg)
        params = agent.get_weights()
        params['fc1.weight'] = params['fc2.bias']
        with self.assertRaises(AssertionError):
            agent.set_weights(params)

    def test_set_weights_with_modified_params(self):
        agent = TestAgent(self.alg)
        params = agent.get_weights()
        params['fc1.weight'][0][0] = 100
        params['fc1.bias'][0] = 100
        params['fc2.weight'][0][0] = 100
        params['fc2.bias'][0] = 100
        params['fc3.weight'][0][0] = 100
        params['fc3.bias'][0] = 100
        agent.set_weights(params)
        new_params = agent.get_weights()
        for i, j in zip(params.values(), new_params.values()):
            self.assertLessEqual(abs(i.sum() - j.sum()), 1e-3)

    def test_train_and_eval_mode(self):
        model = TestModel()
        alg = TestAlgorithm(model)
        agent = TestAgent(alg)
        obs = np.random.random([1, 4]).astype('float32')
        agent.train()
        self.assertTrue(agent.training)
        train_mode_output = agent.predict(obs)
        agent.eval()
        self.assertFalse(agent.training)
        eval_mode_output = agent.predict(obs)
        self.assertEqual(train_mode_output, eval_mode_output)

    def test_train_and_eval_mode_with_dropout(self):
        model = TestModelWithDropout()
        alg = TestAlgorithm(model)
        agent = TestAgent(alg)
        obs = np.random.random([1, 4]).astype('float32')
        agent.train()
        self.assertTrue(agent.training)
        train_mode_output = agent.predict(obs)
        agent.eval()
        self.assertFalse(agent.training)
        eval_mode_output = agent.predict(obs)
        self.assertNotEqual(train_mode_output, eval_mode_output)

    def test_train_and_eval_mode_with_bn(self):
        model = TestModelWithBN()
        alg = TestAlgorithm(model)
        agent = TestAgent(alg)
        obs = np.random.random([1, 4]).astype('float32')
        agent.train()
        self.assertTrue(agent.training)
        train_mode_output = agent.predict(obs)
        agent.eval()
        self.assertFalse(agent.training)
        eval_mode_output = agent.predict(obs)
        self.assertNotEqual(train_mode_output, eval_mode_output)


if __name__ == '__main__':
    unittest.main()
