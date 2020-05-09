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
import parl
import unittest
from copy import deepcopy
from paddle.fluid import ParamAttr
from parl import layers
from parl.utils import get_gpu_count
from parl.core.fluid.plutils import fetch_value


class TestModel(parl.Model):
    def __init__(self):
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
            size=1,
            act=None,
            param_attr=ParamAttr(name='fc3.w'),
            bias_attr=ParamAttr(name='fc3.b'))

    def predict(self, obs):
        out = self.fc1(obs)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


class TestModel2(parl.Model):
    def __init__(self):
        self.created_param = layers.create_parameter(
            shape=[100],
            dtype='float32',
            default_initializer=fluid.initializer.Uniform(low=-1.0, high=1.0))

    def predict(self, obs):
        out = obs + self.created_param()
        return out


class TestModel3(parl.Model):
    def __init__(self):
        self.fc1 = layers.fc(64, bias_attr=False)
        self.batch_norm = layers.batch_norm()

    def predict(self, obs):
        hid1 = self.fc1(obs)
        out = self.batch_norm(hid1)
        return out


class TestModel4(parl.Model):
    def __init__(self):
        self.fc1 = layers.fc(size=256)
        self.fc2 = layers.fc(size=128)
        self.fc3 = layers.fc(size=1)

    def predict(self, obs):
        out = self.fc1(obs)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


class TestModel6(parl.Model):
    def __init__(self):
        self.fc1 = layers.fc(
            size=256,
            act=None,
            param_attr=ParamAttr(name='fc1.w'),
            bias_attr=ParamAttr(name='fc1.b'))
        self.fc_tuple = (layers.fc(
            size=128,
            act=None,
            param_attr=ParamAttr(name='fc2.w'),
            bias_attr=ParamAttr(name='fc2.b')), (layers.fc(
                size=1,
                act=None,
                param_attr=ParamAttr(name='fc3.w'),
                bias_attr=ParamAttr(name='fc3.b')), 10), 10)
        self.fc_dict = {
            'k1':
            layers.fc(
                size=128,
                act=None,
                param_attr=ParamAttr(name='fc4.w'),
                bias_attr=ParamAttr(name='fc4.b')),
            'k2': {
                'k22':
                layers.fc(
                    size=1,
                    act=None,
                    param_attr=ParamAttr(name='fc5.w'),
                    bias_attr=ParamAttr(name='fc5.b'))
            },
            'k3':
            1,
        }


class ModelBaseTest(unittest.TestCase):
    def setUp(self):
        self.model = TestModel()
        self.target_model = deepcopy(self.model)
        self.target_model2 = deepcopy(self.model)

        gpu_count = get_gpu_count()
        if gpu_count > 0:
            place = fluid.CUDAPlace(0)
        else:
            place = fluid.CPUPlace()
        self.executor = fluid.Executor(place)

    def test_model_copy(self):
        self.assertNotEqual(self.model.fc1.param_name,
                            self.target_model.fc1.param_name)
        self.assertNotEqual(self.model.fc1.bias_name,
                            self.target_model.fc1.bias_name)

        self.assertNotEqual(self.model.fc2.param_name,
                            self.target_model.fc2.param_name)
        self.assertNotEqual(self.model.fc2.bias_name,
                            self.target_model.fc2.bias_name)

        self.assertNotEqual(self.model.fc3.param_name,
                            self.target_model.fc3.param_name)
        self.assertNotEqual(self.model.fc3.bias_name,
                            self.target_model.fc3.bias_name)

    def test_model_copy_with_multi_copy(self):
        self.assertNotEqual(self.target_model.fc1.param_name,
                            self.target_model2.fc1.param_name)
        self.assertNotEqual(self.target_model.fc1.bias_name,
                            self.target_model2.fc1.bias_name)

        self.assertNotEqual(self.target_model.fc2.param_name,
                            self.target_model2.fc2.param_name)
        self.assertNotEqual(self.target_model.fc2.bias_name,
                            self.target_model2.fc2.bias_name)

        self.assertNotEqual(self.target_model.fc3.param_name,
                            self.target_model2.fc3.param_name)
        self.assertNotEqual(self.target_model.fc3.bias_name,
                            self.target_model2.fc3.bias_name)

    def test_model_parameters(self):
        self.assertSetEqual(
            set(self.model.parameters()),
            set(['fc1.w', 'fc1.b', 'fc2.w', 'fc2.b', 'fc3.w', 'fc3.b']))

        # Second test for cache parameters
        self.assertSetEqual(
            set(self.model.parameters()),
            set(['fc1.w', 'fc1.b', 'fc2.w', 'fc2.b', 'fc3.w', 'fc3.b']))

        model2 = TestModel6()
        self.assertSetEqual(
            set(model2.parameters()),
            set([
                'fc1.w', 'fc1.b', 'fc2.w', 'fc2.b', 'fc3.w', 'fc3.b', 'fc4.w',
                'fc4.b', 'fc5.w', 'fc5.b'
            ]))

    def test_sync_weights_in_one_program(self):
        pred_program = fluid.Program()
        with fluid.program_guard(pred_program):
            obs = layers.data(name='obs', shape=[4], dtype='float32')
            model_output = self.model.predict(obs)
            target_model_output = self.target_model.predict(obs)
        self.executor.run(fluid.default_startup_program())

        N = 10
        random_obs = np.random.random(size=(N, 4)).astype('float32')
        for i in range(N):
            x = np.expand_dims(random_obs[i], axis=0)
            outputs = self.executor.run(
                pred_program,
                feed={'obs': x},
                fetch_list=[model_output, target_model_output])
            self.assertNotEqual(outputs[0].flatten(), outputs[1].flatten())

        self.model.sync_weights_to(self.target_model)

        random_obs = np.random.random(size=(N, 4)).astype('float32')
        for i in range(N):
            x = np.expand_dims(random_obs[i], axis=0)
            outputs = self.executor.run(
                pred_program,
                feed={'obs': x},
                fetch_list=[model_output, target_model_output])
            self.assertEqual(outputs[0].flatten(), outputs[1].flatten())

    def test_sync_weights_among_programs(self):
        pred_program = fluid.Program()
        pred_program_2 = fluid.Program()
        with fluid.program_guard(pred_program):
            obs = layers.data(name='obs', shape=[4], dtype='float32')
            model_output = self.model.predict(obs)

        # program 2
        with fluid.program_guard(pred_program_2):
            obs = layers.data(name='obs', shape=[4], dtype='float32')
            target_model_output = self.target_model.predict(obs)

        self.executor.run(fluid.default_startup_program())

        N = 10
        random_obs = np.random.random(size=(N, 4)).astype('float32')
        for i in range(N):
            x = np.expand_dims(random_obs[i], axis=0)
            outputs = self.executor.run(
                pred_program, feed={'obs': x}, fetch_list=[model_output])

            outputs_2 = self.executor.run(
                pred_program_2,
                feed={'obs': x},
                fetch_list=[target_model_output])
            self.assertNotEqual(outputs[0].flatten(), outputs_2[0].flatten())

        self.model.sync_weights_to(self.target_model)

        random_obs = np.random.random(size=(N, 4)).astype('float32')
        for i in range(N):
            x = np.expand_dims(random_obs[i], axis=0)
            outputs = self.executor.run(
                pred_program, feed={'obs': x}, fetch_list=[model_output])

            outputs_2 = self.executor.run(
                pred_program_2,
                feed={'obs': x},
                fetch_list=[target_model_output])
            self.assertEqual(outputs[0].flatten(), outputs_2[0].flatten())

    def _numpy_update(self, target_model, decay):
        model_fc1_w = fetch_value('fc1.w')
        model_fc1_b = fetch_value('fc1.b')
        model_fc2_w = fetch_value('fc2.w')
        model_fc2_b = fetch_value('fc2.b')
        model_fc3_w = fetch_value('fc3.w')
        model_fc3_b = fetch_value('fc3.b')

        unique_id = target_model.parameters()[0].split('_')[-1]
        target_model_fc1_w = fetch_value(
            'PARL_target_fc1.w_{}'.format(unique_id))
        target_model_fc1_b = fetch_value(
            'PARL_target_fc1.b_{}'.format(unique_id))
        target_model_fc2_w = fetch_value(
            'PARL_target_fc2.w_{}'.format(unique_id))
        target_model_fc2_b = fetch_value(
            'PARL_target_fc2.b_{}'.format(unique_id))
        target_model_fc3_w = fetch_value(
            'PARL_target_fc3.w_{}'.format(unique_id))
        target_model_fc3_b = fetch_value(
            'PARL_target_fc3.b_{}'.format(unique_id))

        # updated self.target_model parameters value in numpy way
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

        return (target_model_fc1_w, target_model_fc1_b, target_model_fc2_w,
                target_model_fc2_b, target_model_fc3_w, target_model_fc3_b)

    def test_sync_weights_with_decay(self):
        pred_program = fluid.Program()
        with fluid.program_guard(pred_program):
            obs = layers.data(name='obs', shape=[4], dtype='float32')
            model_output = self.model.predict(obs)
            target_model_output = self.target_model.predict(obs)

        self.executor.run(fluid.default_startup_program())

        decay = 0.9
        # update in numpy way
        (target_model_fc1_w, target_model_fc1_b, target_model_fc2_w,
         target_model_fc2_b, target_model_fc3_w,
         target_model_fc3_b) = self._numpy_update(self.target_model, decay)

        self.model.sync_weights_to(self.target_model, decay=decay)

        N = 10
        random_obs = np.random.random(size=(N, 4)).astype('float32')
        for i in range(N):
            x = np.expand_dims(random_obs[i], axis=0)
            real_target_outputs = self.executor.run(
                pred_program,
                feed={'obs': x},
                fetch_list=[target_model_output])[0]

            # Ideal target output
            out_np = np.dot(x, target_model_fc1_w) + target_model_fc1_b
            out_np = np.dot(out_np, target_model_fc2_w) + target_model_fc2_b
            out_np = np.dot(out_np, target_model_fc3_w) + target_model_fc3_b

            self.assertLess(float(np.abs(real_target_outputs - out_np)), 1e-5)

    def test_sync_weights_with_decay_with_multi_sync(self):
        pred_program = fluid.Program()
        with fluid.program_guard(pred_program):
            obs = layers.data(name='obs', shape=[4], dtype='float32')
            model_output = self.model.predict(obs)
            target_model_output = self.target_model.predict(obs)

        self.executor.run(fluid.default_startup_program())

        decay = 0.9
        # update in numpy way
        (target_model_fc1_w, target_model_fc1_b, target_model_fc2_w,
         target_model_fc2_b, target_model_fc3_w,
         target_model_fc3_b) = self._numpy_update(self.target_model, decay)

        self.model.sync_weights_to(self.target_model, decay=decay)

        N = 10
        random_obs = np.random.random(size=(N, 4)).astype('float32')
        for i in range(N):
            x = np.expand_dims(random_obs[i], axis=0)
            real_target_outputs = self.executor.run(
                pred_program,
                feed={'obs': x},
                fetch_list=[target_model_output])[0]

            # Ideal target output
            out_np = np.dot(x, target_model_fc1_w) + target_model_fc1_b
            out_np = np.dot(out_np, target_model_fc2_w) + target_model_fc2_b
            out_np = np.dot(out_np, target_model_fc3_w) + target_model_fc3_b

            self.assertLess(float(np.abs(real_target_outputs - out_np)), 1e-5)

        decay = 0.9
        # update in numpy way
        (target_model_fc1_w, target_model_fc1_b, target_model_fc2_w,
         target_model_fc2_b, target_model_fc3_w,
         target_model_fc3_b) = self._numpy_update(self.target_model, decay)

        self.model.sync_weights_to(self.target_model, decay=decay)

        N = 10
        random_obs = np.random.random(size=(N, 4)).astype('float32')
        for i in range(N):
            x = np.expand_dims(random_obs[i], axis=0)
            real_target_outputs = self.executor.run(
                pred_program,
                feed={'obs': x},
                fetch_list=[target_model_output])[0]

            # Ideal target output
            out_np = np.dot(x, target_model_fc1_w) + target_model_fc1_b
            out_np = np.dot(out_np, target_model_fc2_w) + target_model_fc2_b
            out_np = np.dot(out_np, target_model_fc3_w) + target_model_fc3_b

            self.assertLess(float(np.abs(real_target_outputs - out_np)), 1e-5)

    def test_sync_weights_with_different_decay(self):
        pred_program = fluid.Program()
        with fluid.program_guard(pred_program):
            obs = layers.data(name='obs', shape=[4], dtype='float32')
            model_output = self.model.predict(obs)
            target_model_output = self.target_model.predict(obs)

        self.executor.run(fluid.default_startup_program())

        decay = 0.9
        # update in numpy way
        (target_model_fc1_w, target_model_fc1_b, target_model_fc2_w,
         target_model_fc2_b, target_model_fc3_w,
         target_model_fc3_b) = self._numpy_update(self.target_model, decay)

        self.model.sync_weights_to(self.target_model, decay=decay)

        N = 10
        random_obs = np.random.random(size=(N, 4)).astype('float32')
        for i in range(N):
            x = np.expand_dims(random_obs[i], axis=0)
            real_target_outputs = self.executor.run(
                pred_program,
                feed={'obs': x},
                fetch_list=[target_model_output])[0]

            # Ideal target output
            out_np = np.dot(x, target_model_fc1_w) + target_model_fc1_b
            out_np = np.dot(out_np, target_model_fc2_w) + target_model_fc2_b
            out_np = np.dot(out_np, target_model_fc3_w) + target_model_fc3_b

            self.assertLess(float(np.abs(real_target_outputs - out_np)), 1e-5)

        decay = 0.8
        # update in numpy way
        (target_model_fc1_w, target_model_fc1_b, target_model_fc2_w,
         target_model_fc2_b, target_model_fc3_w,
         target_model_fc3_b) = self._numpy_update(self.target_model, decay)

        self.model.sync_weights_to(self.target_model, decay=decay)

        N = 10
        random_obs = np.random.random(size=(N, 4)).astype('float32')
        for i in range(N):
            x = np.expand_dims(random_obs[i], axis=0)
            real_target_outputs = self.executor.run(
                pred_program,
                feed={'obs': x},
                fetch_list=[target_model_output])[0]

            # Ideal target output
            out_np = np.dot(x, target_model_fc1_w) + target_model_fc1_b
            out_np = np.dot(out_np, target_model_fc2_w) + target_model_fc2_b
            out_np = np.dot(out_np, target_model_fc3_w) + target_model_fc3_b

            self.assertLess(float(np.abs(real_target_outputs - out_np)), 1e-5)

    def test_sync_weights_with_multi_target_model(self):
        pred_program = fluid.Program()
        with fluid.program_guard(pred_program):
            obs = layers.data(name='obs', shape=[4], dtype='float32')
            model_output = self.model.predict(obs)
            target_model_output = self.target_model.predict(obs)
            target_model_output2 = self.target_model2.predict(obs)

        self.executor.run(fluid.default_startup_program())

        decay = 0.9
        # update in numpy way
        (target_model_fc1_w, target_model_fc1_b, target_model_fc2_w,
         target_model_fc2_b, target_model_fc3_w,
         target_model_fc3_b) = self._numpy_update(self.target_model, decay)

        self.model.sync_weights_to(self.target_model, decay=decay)

        N = 10
        random_obs = np.random.random(size=(N, 4)).astype('float32')
        for i in range(N):
            x = np.expand_dims(random_obs[i], axis=0)
            real_target_outputs = self.executor.run(
                pred_program,
                feed={'obs': x},
                fetch_list=[target_model_output])[0]

            # Ideal target output
            out_np = np.dot(x, target_model_fc1_w) + target_model_fc1_b
            out_np = np.dot(out_np, target_model_fc2_w) + target_model_fc2_b
            out_np = np.dot(out_np, target_model_fc3_w) + target_model_fc3_b

            self.assertLess(float(np.abs(real_target_outputs - out_np)), 1e-5)

        decay = 0.8
        # update in numpy way
        (target_model_fc1_w, target_model_fc1_b, target_model_fc2_w,
         target_model_fc2_b, target_model_fc3_w,
         target_model_fc3_b) = self._numpy_update(self.target_model2, decay)

        self.model.sync_weights_to(self.target_model2, decay=decay)

        N = 10
        random_obs = np.random.random(size=(N, 4)).astype('float32')
        for i in range(N):
            x = np.expand_dims(random_obs[i], axis=0)
            real_target_outputs = self.executor.run(
                pred_program,
                feed={'obs': x},
                fetch_list=[target_model_output2])[0]

            # Ideal target output
            out_np = np.dot(x, target_model_fc1_w) + target_model_fc1_b
            out_np = np.dot(out_np, target_model_fc2_w) + target_model_fc2_b
            out_np = np.dot(out_np, target_model_fc3_w) + target_model_fc3_b

            self.assertLess(float(np.abs(real_target_outputs - out_np)), 1e-5)

    def test_sync_weights_with_create_parameter(self):
        model = TestModel2()
        target_model = deepcopy(model)

        pred_program = fluid.Program()
        with fluid.program_guard(pred_program):
            obs = layers.data(name='obs', shape=[100], dtype='float32')
            model_output = model.predict(obs)
            target_model_output = target_model.predict(obs)
        self.executor.run(fluid.default_startup_program())

        N = 10
        random_obs = np.random.random(size=(N, 100)).astype('float32')
        for i in range(N):
            x = np.expand_dims(random_obs[i], axis=0)
            outputs = self.executor.run(
                pred_program,
                feed={'obs': x},
                fetch_list=[model_output, target_model_output])
            self.assertNotEqual(
                np.sum(outputs[0].flatten()), np.sum(outputs[1].flatten()))

        model.sync_weights_to(target_model)

        random_obs = np.random.random(size=(N, 100)).astype('float32')
        for i in range(N):
            x = np.expand_dims(random_obs[i], axis=0)
            outputs = self.executor.run(
                pred_program,
                feed={'obs': x},
                fetch_list=[model_output, target_model_output])
            self.assertEqual(
                np.sum(outputs[0].flatten()), np.sum(outputs[1].flatten()))

    def test_sync_weights_with_batch_norm(self):
        model = TestModel3()
        target_model = deepcopy(model)

        program1 = fluid.Program()
        program2 = fluid.Program()
        with fluid.program_guard(program1):
            obs = layers.data(
                name='obs', shape=[32, 128, 128], dtype="float32")
            model_output = model.predict(obs)
            loss = layers.reduce_mean(model_output)
            optimizer = fluid.optimizer.AdamOptimizer(1e-3)
            optimizer.minimize(loss)

        with fluid.program_guard(program2):
            obs = layers.data(
                name='obs', shape=[32, 128, 128], dtype="float32")
            model_output = model.predict(obs)
            target_model_output = target_model.predict(obs)
        self.executor.run(fluid.default_startup_program())

        N = 10
        random_obs = np.random.random(size=(N, 32, 128, 128)).astype('float32')
        for i in range(N):
            x = np.expand_dims(random_obs[i], axis=0)
            outputs = self.executor.run(
                program2,
                feed={'obs': x},
                fetch_list=[model_output, target_model_output])
            self.assertNotEqual(
                np.sum(outputs[0].flatten()), np.sum(outputs[1].flatten()))

        # run optimizing to make parameters of batch_norm between model and target_model are different
        N = 100
        random_obs = np.random.random(size=(N, 32, 128, 128)).astype('float32')
        for i in range(N):
            x = np.expand_dims(random_obs[i], axis=0)
            self.executor.run(program1, feed={'obs': x})

        model.sync_weights_to(target_model)

        random_obs = np.random.random(size=(N, 32, 128, 128)).astype('float32')
        for i in range(N):
            x = np.expand_dims(random_obs[i], axis=0)
            outputs = self.executor.run(
                program2,
                feed={'obs': x},
                fetch_list=[model_output, target_model_output])
            self.assertEqual(
                np.sum(outputs[0].flatten()), np.sum(outputs[1].flatten()))

    def test_get_weights(self):
        pred_program = fluid.Program()
        with fluid.program_guard(pred_program):
            obs = layers.data(name='obs', shape=[4], dtype='float32')
            model_output = self.model.predict(obs)

        self.executor.run(fluid.default_startup_program())

        expected_params = []
        for param_name in [
                'fc1.w', 'fc1.b', 'fc2.w', 'fc2.b', 'fc3.w', 'fc3.b'
        ]:
            expected_params.append(fetch_value(param_name))

        params = self.model.get_weights()
        self.assertEqual(len(params), len(expected_params))
        for param in params:
            flag = False
            for expected_param in expected_params:
                if np.sum(param) - np.sum(expected_param) < 1e-5:
                    flag = True
                    break
            self.assertTrue(flag)

    def test_set_weights(self):
        pred_program = fluid.Program()
        with fluid.program_guard(pred_program):
            obs = layers.data(name='obs', shape=[4], dtype='float32')
            model_output = self.model.predict(obs)

        self.executor.run(fluid.default_startup_program())

        params = self.model.get_weights()
        new_params = [x + 1.0 for x in params]

        self.model.set_weights(new_params)

        for x, y in list(zip(new_params, self.model.get_weights())):
            self.assertEqual(np.sum(x), np.sum(y))

    def test_set_weights_between_different_models(self):
        model1 = TestModel4()
        model2 = TestModel4()

        pred_program = fluid.Program()
        with fluid.program_guard(pred_program):
            obs = layers.data(name='obs', shape=[4], dtype='float32')
            model1_output = model1.predict(obs)
            model2_output = model2.predict(obs)

        self.executor.run(fluid.default_startup_program())

        N = 10
        random_obs = np.random.random(size=(N, 4)).astype('float32')
        for i in range(N):
            x = np.expand_dims(random_obs[i], axis=0)
            outputs = self.executor.run(
                pred_program,
                feed={'obs': x},
                fetch_list=[model1_output, model2_output])
            self.assertNotEqual(outputs[0].flatten(), outputs[1].flatten())

        # pass parameters of self.model to model2
        params = model1.get_weights()
        model2.set_weights(params)

        random_obs = np.random.random(size=(N, 4)).astype('float32')
        for i in range(N):
            x = np.expand_dims(random_obs[i], axis=0)
            outputs = self.executor.run(
                pred_program,
                feed={'obs': x},
                fetch_list=[model1_output, model2_output])
            self.assertEqual(outputs[0].flatten(), outputs[1].flatten())

    def test_set_weights_with_wrong_params_num(self):
        pred_program = fluid.Program()
        with fluid.program_guard(pred_program):
            obs = layers.data(name='obs', shape=[4], dtype='float32')
            model_output = self.model.predict(obs)

        self.executor.run(fluid.default_startup_program())

        params = self.model.get_weights()

        with self.assertRaises(AssertionError):
            self.model.set_weights(params[1:])

    def test_set_weights_with_wrong_params_shape(self):
        pred_program = fluid.Program()
        with fluid.program_guard(pred_program):
            obs = layers.data(name='obs', shape=[4], dtype='float32')
            model_output = self.model.predict(obs)

        self.executor.run(fluid.default_startup_program())

        params = self.model.get_weights()

        params.reverse()

        self.model.set_weights(params)

        x = np.random.random(size=(1, 4)).astype('float32')

        with self.assertRaises(fluid.core_avx.EnforceNotMet):
            self.executor.run(
                pred_program, feed={'obs': x}, fetch_list=[model_output])

    def test_get_weights_set_weights_with_create_parameter(self):
        model1 = TestModel2()
        model2 = TestModel2()

        pred_program = fluid.Program()
        with fluid.program_guard(pred_program):
            obs = layers.data(name='obs', shape=[100], dtype='float32')
            model1_output = model1.predict(obs)
            model2_output = model2.predict(obs)

        self.executor.run(fluid.default_startup_program())

        N = 10
        random_obs = np.random.random(size=(N, 100)).astype('float32')
        for i in range(N):
            x = np.expand_dims(random_obs[i], axis=0)
            outputs = self.executor.run(
                pred_program,
                feed={'obs': x},
                fetch_list=[model1_output, model2_output])
            self.assertNotEqual(
                np.sum(outputs[0].flatten()), np.sum(outputs[1].flatten()))

        # pass parameters of self.model to model2
        params = model1.get_weights()
        model2.set_weights(params)

        random_obs = np.random.random(size=(N, 100)).astype('float32')
        for i in range(N):
            x = np.expand_dims(random_obs[i], axis=0)
            outputs = self.executor.run(
                pred_program,
                feed={'obs': x},
                fetch_list=[model1_output, model2_output])
            self.assertEqual(
                np.sum(outputs[0].flatten()), np.sum(outputs[1].flatten()))


if __name__ == '__main__':
    unittest.main()
