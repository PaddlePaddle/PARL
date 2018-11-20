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
from parl.framework.model_base import Model
from copy import deepcopy
import unittest

class Value(Model):
    def __init__(self, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        self.fc1 = layers.fc(size=256, act='relu')
        self.fc2 = layers.fc(size=128, act='relu')

class ModelBaseTest(unittest.TestCase):
    def test_network_copy(self):
        value = Value(obs_dim=2, act_dim=1)
        target_value = deepcopy(value)
        self.assertNotEqual(value.fc1.param_name, target_value.fc1.param_name)
        self.assertNotEqual(value.fc1.bias_name, target_value.fc1.bias_name)

        self.assertNotEqual(value.fc2.param_name, target_value.fc2.param_name)
        self.assertNotEqual(value.fc2.param_name, target_value.fc2.param_name)

if __name__ == '__main__':
    unittest.main()
