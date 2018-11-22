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
import unittest
from copy import deepcopy
from paddle.fluid import ParamAttr
from parl.framework.model_base import Model


class Value(Model):
    def __init__(self, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.fc1 = layers.fc(
            size=256,
            act='relu',
            param_attr=ParamAttr(name='fc1.w'),
            bias_attr=ParamAttr(name='fc1.b'))
        self.fc2 = layers.fc(
            size=128,
            act='relu',
            param_attr=ParamAttr(name='fc2.w'),
            bias_attr=ParamAttr(name='fc2.b'))


class ModelBaseTest(unittest.TestCase):
    def test_network_copy(self):
        value = Value(obs_dim=2, act_dim=1)
        target_value = deepcopy(value)
        self.assertNotEqual(value.fc1.param_name, target_value.fc1.param_name)
        self.assertNotEqual(value.fc1.bias_name, target_value.fc1.bias_name)

        self.assertNotEqual(value.fc2.param_name, target_value.fc2.param_name)
        self.assertNotEqual(value.fc2.param_name, target_value.fc2.param_name)

    def test_network_copy_with_multi_copy(self):
        value = Value(obs_dim=2, act_dim=1)
        target_value1 = deepcopy(value)
        target_value2 = deepcopy(value)
        self.assertNotEqual(target_value1.fc1.param_name,
                            target_value2.fc1.param_name)
        self.assertNotEqual(target_value1.fc1.bias_name,
                            target_value2.fc1.bias_name)

        self.assertNotEqual(target_value1.fc2.param_name,
                            target_value2.fc2.param_name)
        self.assertNotEqual(target_value1.fc2.param_name,
                            target_value2.fc2.param_name)

    def test_network_parameter_names(self):
        value = Value(obs_dim=2, act_dim=2)
        parameter_names = value.parameter_names
        self.assertSetEqual(
            set(parameter_names), set(['fc1.w', 'fc1.b', 'fc2.w', 'fc2.b']))


if __name__ == '__main__':
    unittest.main()
