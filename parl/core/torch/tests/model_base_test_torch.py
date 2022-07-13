#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim

from parl.utils import get_gpu_count
import parl


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


class ModelBaseTest(unittest.TestCase):
    def setUp(self):
        self.model = TestModel()
        self.target_model = TestModel()
        self.target_model2 = TestModel()
        self.target_model3 = TestModel()

        gpu_count = get_gpu_count()
        device = torch.device('cuda' if gpu_count else 'cpu')

    def test_sync_weights_in_one_program(self):
        obs = torch.randn(1, 4)

        N = 10
        random_obs = torch.randn(N, 4)
        for i in range(N):
            x = random_obs[i].view(1, -1)
            model_output = self.model.predict(x).item()
            target_model_output = self.target_model.predict(x).item()
            self.assertNotEqual(model_output, target_model_output)

        self.model.sync_weights_to(self.target_model)

        random_obs = torch.randn(N, 4)
        for i in range(N):
            x = random_obs[i].view(1, -1)
            model_output = self.model.predict(x).item()
            target_model_output = self.target_model.predict(x).item()
            self.assertEqual(model_output, target_model_output)

    def _numpy_update(self, target_model, decay):
        target_parameters = dict(target_model.named_parameters())
        updated_parameters = {}
        for name, param in self.model.named_parameters():
            updated_parameters[name] = decay * target_parameters[name].detach(
            ).cpu().numpy() + (1 - decay) * param.detach().cpu().numpy()
        return updated_parameters

    def test_sync_weights_with_decay(self):
        decay = 0.9
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
        random_obs = np.random.randn(N, 4)
        for i in range(N):
            obs = np.expand_dims(random_obs[i], -1)  # 4, 1
            real_target_outputs = self.target_model.predict(
                torch.Tensor(obs).view(1, -1)).item()

            out_np = np.dot(target_model_fc1_w, obs) + np.expand_dims(
                target_model_fc1_b, -1)  # (256, 256)
            out_np = np.dot(target_model_fc2_w, out_np) + np.expand_dims(
                target_model_fc2_b, -1)
            out_np = np.dot(target_model_fc3_w, out_np) + np.expand_dims(
                target_model_fc3_b, -1)

            self.assertLess(float(np.abs(real_target_outputs - out_np)), 1e-5)

    def test_sync_weights_with_multi_decay(self):
        decay = 0.9
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
        random_obs = np.random.randn(N, 4)
        for i in range(N):
            obs = np.expand_dims(random_obs[i], -1)  # 4, 1
            real_target_outputs = self.target_model.predict(
                torch.Tensor(obs).view(1, -1)).item()

            out_np = np.dot(target_model_fc1_w, obs) + np.expand_dims(
                target_model_fc1_b, -1)  # (256, 256)
            out_np = np.dot(target_model_fc2_w, out_np) + np.expand_dims(
                target_model_fc2_b, -1)
            out_np = np.dot(target_model_fc3_w, out_np) + np.expand_dims(
                target_model_fc3_b, -1)

            self.assertLess(float(np.abs(real_target_outputs - out_np)), 1e-5)

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
        random_obs = np.random.randn(N, 4)
        for i in range(N):
            obs = np.expand_dims(random_obs[i], -1)  # 4, 1
            real_target_outputs = self.target_model.predict(
                torch.Tensor(obs).view(1, -1)).item()

            out_np = np.dot(target_model_fc1_w, obs) + np.expand_dims(
                target_model_fc1_b, -1)  # (256, 256)
            out_np = np.dot(target_model_fc2_w, out_np) + np.expand_dims(
                target_model_fc2_b, -1)
            out_np = np.dot(target_model_fc3_w, out_np) + np.expand_dims(
                target_model_fc3_b, -1)

            self.assertLess(float(np.abs(real_target_outputs - out_np)), 1e-5)

    def test_sync_weights_with_different_decay(self):
        decay = 0.9
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
        random_obs = np.random.randn(N, 4)
        for i in range(N):
            obs = np.expand_dims(random_obs[i], -1)  # 4, 1
            real_target_outputs = self.target_model.predict(
                torch.Tensor(obs).view(1, -1)).item()

            out_np = np.dot(target_model_fc1_w, obs) + np.expand_dims(
                target_model_fc1_b, -1)  # (256, 256)
            out_np = np.dot(target_model_fc2_w, out_np) + np.expand_dims(
                target_model_fc2_b, -1)
            out_np = np.dot(target_model_fc3_w, out_np) + np.expand_dims(
                target_model_fc3_b, -1)

            self.assertLess(float(np.abs(real_target_outputs - out_np)), 1e-5)

        decay = 0.8

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
        random_obs = np.random.randn(N, 4)
        for i in range(N):
            obs = np.expand_dims(random_obs[i], -1)  # 4, 1
            real_target_outputs = self.target_model.predict(
                torch.Tensor(obs).view(1, -1)).item()

            out_np = np.dot(target_model_fc1_w, obs) + np.expand_dims(
                target_model_fc1_b, -1)  # (256, 256)
            out_np = np.dot(target_model_fc2_w, out_np) + np.expand_dims(
                target_model_fc2_b, -1)
            out_np = np.dot(target_model_fc3_w, out_np) + np.expand_dims(
                target_model_fc3_b, -1)

            self.assertLess(float(np.abs(real_target_outputs - out_np)), 1e-5)

    def test_sync_weights_with_different_target_model(self):
        decay = 0.9
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
        random_obs = np.random.randn(N, 4)
        for i in range(N):
            obs = np.expand_dims(random_obs[i], -1)  # 4, 1
            real_target_outputs = self.target_model.predict(
                torch.Tensor(obs).view(1, -1)).item()

            out_np = np.dot(target_model_fc1_w, obs) + np.expand_dims(
                target_model_fc1_b, -1)  # (256, 256)
            out_np = np.dot(target_model_fc2_w, out_np) + np.expand_dims(
                target_model_fc2_b, -1)
            out_np = np.dot(target_model_fc3_w, out_np) + np.expand_dims(
                target_model_fc3_b, -1)

            self.assertLess(float(np.abs(real_target_outputs - out_np)), 1e-5)

        decay = 0.8

        updated_parameters = self._numpy_update(self.target_model2, decay)
        (target_model_fc1_w, target_model_fc1_b, target_model_fc2_w,
         target_model_fc2_b, target_model_fc3_w,
         target_model_fc3_b) = (updated_parameters['fc1.weight'],
                                updated_parameters['fc1.bias'],
                                updated_parameters['fc2.weight'],
                                updated_parameters['fc2.bias'],
                                updated_parameters['fc3.weight'],
                                updated_parameters['fc3.bias'])

        self.model.sync_weights_to(self.target_model2, decay)
        random_obs = np.random.randn(N, 4)
        for i in range(N):
            obs = np.expand_dims(random_obs[i], -1)  # 4, 1
            real_target_outputs = self.target_model2.predict(
                torch.Tensor(obs).view(1, -1)).item()

            out_np = np.dot(target_model_fc1_w, obs) + np.expand_dims(
                target_model_fc1_b, -1)  # (256, 256)
            out_np = np.dot(target_model_fc2_w, out_np) + np.expand_dims(
                target_model_fc2_b, -1)
            out_np = np.dot(target_model_fc3_w, out_np) + np.expand_dims(
                target_model_fc3_b, -1)

            self.assertLess(float(np.abs(real_target_outputs - out_np)), 1e-5)

    def test_get_weights(self):
        params = self.model.get_weights()
        expected_params = list(self.model.parameters())
        self.assertEqual(len(params), len(expected_params))
        for i, key in enumerate(params):
            self.assertLess(
                (params[key].sum().item() - expected_params[i].sum().item()),
                1e-5)

    def test_set_weights(self):
        params = self.model.get_weights()
        self.target_model3.set_weights(params)

        for i, j in zip(params.values(),
                        self.target_model3.get_weights().values()):
            self.assertLessEqual(abs(i.sum().item() - j.sum().item()), 1e-3)

    def test_set_weights_between_different_models(self):
        model1 = TestModel()
        model2 = TestModel()

        N = 10
        random_obs = torch.randn(N, 4)
        for i in range(N):
            x = random_obs[i].view(1, -1)
            model1_output = model1.predict(x).item()
            model2_output = model2.predict(x).item()
            self.assertNotEqual(model1_output, model2_output)

        params = model1.get_weights()
        model2.set_weights(params)

        random_obs = torch.randn(N, 4)
        for i in range(N):
            x = random_obs[i].view(1, -1)
            model1_output = model1.predict(x).item()
            model2_output = model2.predict(x).item()
            self.assertEqual(model1_output, model2_output)

    def test_set_weights_wrong_params_num(self):
        params = self.model.get_weights()
        with self.assertRaises(TypeError):
            self.model.set_weights(params[1:])

    def test_set_weights_wrong_params_shape(self):
        params = self.model.get_weights()
        params['fc1.weight'] = params['fc2.bias']
        with self.assertRaises(RuntimeError):
            self.model.set_weights(params)


if __name__ == '__main__':
    unittest.main()
