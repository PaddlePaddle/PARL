#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn.functional as F
import parl


class MujocoModel(parl.Model):
    def __init__(self, obs_dim, act_dim):
        super(MujocoModel, self).__init__()

        hid1_size = 256
        hid2_size = 256

        value1 = np.sqrt(1.0 / obs_dim)
        value2 = np.sqrt(1.0 / hid1_size)
        value3 = np.sqrt(1.0 / hid2_size)

        param_attr1 = paddle.framework.ParamAttr(
            initializer=paddle.nn.initializer.Uniform(
                low=-value1, high=value1))
        param_attr2 = paddle.framework.ParamAttr(
            initializer=paddle.nn.initializer.Uniform(
                low=-value2, high=value2))
        param_attr3 = paddle.framework.ParamAttr(
            initializer=paddle.nn.initializer.Uniform(
                low=-value3, high=value3))

        self.fc1 = nn.Linear(
            obs_dim, hid1_size, weight_attr=param_attr1, bias_attr=param_attr1)
        self.fc2 = nn.Linear(
            hid1_size,
            hid2_size,
            weight_attr=param_attr2,
            bias_attr=param_attr2)
        self.fc3 = nn.Linear(
            hid2_size, act_dim, weight_attr=param_attr3, bias_attr=param_attr3)

    def forward(self, obs):
        hid1 = F.tanh(self.fc1(obs))
        hid2 = F.tanh(self.fc2(hid1))
        means = self.fc3(hid2)
        return means
