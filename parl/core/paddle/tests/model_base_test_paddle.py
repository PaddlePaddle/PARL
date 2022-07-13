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
import paddle
import paddle.nn as nn
import parl
import unittest
from copy import deepcopy
from parl.core.paddle.model import Model


class TestModel(Model):
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


class ModelBaseTest(unittest.TestCase):
    def setUp(self):
        self.model = TestModel()
        self.target_model = TestModel()
        self.target_model2 = deepcopy(self.model)
        self.target_model3 = deepcopy(self.model)

    def test_model_copy(self):
        self.assertEqual(
            len(self.model.state_dict()), len(self.target_model2.state_dict()))
        for (name1, var1), (name2, var2) in zip(
                self.model.named_parameters(),
                self.target_model2.named_parameters()):
            self.assertEqual(name1, name2)
            var1_np = var1.numpy().sum()
            var2_np = var2.numpy().sum()
            self.assertLess(float(np.abs(var1_np - var2_np)), 1e-5)

    def test_model_copy_with_multi_copy(self):
        self.assertEqual(
            len(self.target_model2.state_dict()),
            len(self.target_model2.state_dict()))
        for (name1, var1), (name2, var2) in zip(
                self.target_model2.named_parameters(),
                self.target_model3.named_parameters()):
            self.assertEqual(name1, name2)
            var1_np = var1.numpy().sum()
            var2_np = var2.numpy().sum()
            self.assertLess(float(np.abs(var1_np - var2_np)), 1e-5)

    def test_sync_weights_in_one_model(self):
        N = 10
        random_obs = paddle.to_tensor(np.random.rand(N, 4).astype(np.float32))
        for i in range(N):
            x = random_obs[i]
            model_output = self.model.predict(x)
            target_model_output = self.target_model.predict(x)
            self.assertNotEqual(model_output, target_model_output)

        self.model.sync_weights_to(self.target_model)

        random_obs = paddle.to_tensor(np.random.rand(N, 4).astype(np.float32))
        for i in range(N):
            x = random_obs[i]
            model_output = self.model.predict(x).numpy()
            target_model_output = self.target_model.predict(x).numpy()
            self.assertEqual(model_output, target_model_output)

    def _numpy_update(self, target_model, decay):
        target_parameters = dict(target_model.named_parameters())
        updated_parameters = {}
        for name, param in self.model.named_parameters():
            updated_parameters[name] = decay * target_parameters[name].detach(
            ).numpy() + (1 - decay) * param.detach().numpy()
        return updated_parameters

    def test_sync_weights_with_different_decay(self):
        decay_list = [0., 0.47, 0.9, 1.0]
        for decay in decay_list:
            updated_parameters = self._numpy_update(self.target_model, decay)
            (target_model_fc1_w, target_model_fc1_b, target_model_fc2_w,
             target_model_fc2_b, target_model_fc3_w,
             target_model_fc3_b) = (updated_parameters['fc1.weight'],
                                    updated_parameters['fc1.bias'],
                                    updated_parameters['fc2.weight'],
                                    updated_parameters['fc2.bias'],
                                    updated_parameters['fc3.weight'],
                                    updated_parameters['fc3.bias'])

            self.model.sync_weights_to(self.target_model, decay)
            N = 10
            random_obs = np.random.randn(N, 4).astype(np.float32)
            for i in range(N):
                obs = random_obs[i]
                real_target_outputs = self.target_model.predict(
                    paddle.to_tensor(obs)).numpy().sum()
                out_np = np.dot(obs, target_model_fc1_w) + target_model_fc1_b
                out_np = np.dot(out_np,
                                target_model_fc2_w) + target_model_fc2_b
                out_np = np.dot(out_np,
                                target_model_fc3_w) + target_model_fc3_b
                out_np = out_np.sum()

                self.assertLess(
                    float(np.abs(real_target_outputs - out_np)), 1e-5)

    def test_sync_weights_with_different_target_model(self):
        decay = 0.9
        for the_target_model in [self.target_model, self.target_model2]:
            updated_parameters = self._numpy_update(the_target_model, decay)
            (target_model_fc1_w, target_model_fc1_b, target_model_fc2_w,
             target_model_fc2_b, target_model_fc3_w,
             target_model_fc3_b) = (updated_parameters['fc1.weight'],
                                    updated_parameters['fc1.bias'],
                                    updated_parameters['fc2.weight'],
                                    updated_parameters['fc2.bias'],
                                    updated_parameters['fc3.weight'],
                                    updated_parameters['fc3.bias'])

            self.model.sync_weights_to(the_target_model, decay)
            N = 10
            random_obs = np.random.randn(N, 4).astype(np.float32)
            for i in range(N):
                obs = random_obs[i]
                real_target_outputs = the_target_model.predict(
                    paddle.to_tensor(obs)).numpy().sum()
                out_np = np.dot(obs, target_model_fc1_w) + target_model_fc1_b
                out_np = np.dot(out_np,
                                target_model_fc2_w) + target_model_fc2_b
                out_np = np.dot(out_np,
                                target_model_fc3_w) + target_model_fc3_b
                out_np = out_np.sum()

                self.assertLess(
                    float(np.abs(real_target_outputs - out_np)), 1e-5)

    def test_get_weights(self):
        params = self.model.get_weights()
        expected_params = list(self.model.named_parameters())
        self.assertEqual(len(params), len(expected_params))
        for i, key in enumerate(params):
            self.assertLess(
                (params[key].sum() - expected_params[i][1].numpy().sum()),
                1e-5)

    def test_set_weights(self):
        params = self.model.get_weights()
        self.target_model3.set_weights(params)

        for i, j in zip(params.values(),
                        self.target_model3.get_weights().values()):
            self.assertLessEqual(abs(i.sum() - j.sum()), 1e-3)

    def test_set_weights_between_different_models(self):
        model1 = TestModel()
        model2 = TestModel()

        N = 10
        random_obs = np.random.randn(N, 4).astype(np.float32)
        for i in range(N):
            x = paddle.to_tensor(random_obs[i])
            model1_output = model1.predict(x).numpy().sum()
            model2_output = model2.predict(x).numpy().sum()
            self.assertNotEqual(model1_output, model2_output)

        params = model1.get_weights()
        model2.set_weights(params)

        random_obs = np.random.randn(N, 4).astype(np.float32)
        for i in range(N):
            x = paddle.to_tensor(random_obs[i])
            model1_output = model1.predict(x).numpy().sum()
            model2_output = model2.predict(x).numpy().sum()
            self.assertEqual(model1_output, model2_output)

    def test_set_weights_with_wrong_params_num(self):
        params = self.model.get_weights()
        del params['fc2.bias']
        with self.assertRaises(AssertionError):
            self.model.set_weights(params)

    def test_set_weights_with_wrong_params_shape(self):
        params = self.model.get_weights()
        params['fc1.weight'] = params['fc2.bias']
        with self.assertRaises(AssertionError):
            self.model.set_weights(params)

    def test_set_weights_with_modified_params(self):
        params = self.model.get_weights()
        params['fc1.weight'][0][0] = 100
        params['fc1.bias'][0] = 100
        params['fc2.weight'][0][0] = 100
        params['fc2.bias'][0] = 100
        params['fc3.weight'][0][0] = 100
        params['fc3.bias'][0] = 100
        self.model.set_weights(params)
        new_params = self.model.get_weights()
        for i, j in zip(params.values(), new_params.values()):
            self.assertLessEqual(abs(i.sum() - j.sum()), 1e-3)


if __name__ == '__main__':
    unittest.main()
