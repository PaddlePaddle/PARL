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

import parl
import paddle
import paddle.nn as nn
import numpy as np


class MujocoModel(parl.Model):
    def __init__(self, obs_space, act_space):
        super(MujocoModel, self).__init__()
        self.continuous_action = True

        self.fc_v1 = nn.Linear(obs_space.shape[0], 64)
        self.fc_v2 = nn.Linear(64, 64)
        self.fc_v3 = nn.Linear(64, 1)

        self.fc_pi1 = nn.Linear(obs_space.shape[0], 64)
        self.fc_pi2 = nn.Linear(64, 64)
        self.fc_pi3 = nn.Linear(64, np.prod(act_space.shape))

        self.fc_pi_std = paddle.static.create_parameter(
            [1, np.prod(act_space.shape)],
            dtype='float32',
            default_initializer=nn.initializer.Constant(value=0))

    def value(self, obs):
        out = paddle.tanh(self.fc_v1(obs))
        out = paddle.tanh(self.fc_v2(out))
        value = self.fc_v3(out)
        return value

    def policy(self, obs):
        out = paddle.tanh(self.fc_pi1(obs))
        out = paddle.tanh(self.fc_pi2(out))
        action_mean = self.fc_pi3(out)

        action_logstd = self.fc_pi_std
        action_std = paddle.exp(action_logstd)
        return action_mean, action_std
