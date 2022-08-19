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
    """ The Model for Mujoco env
    Args:
        obs_space (Box): observation space.
        act_space (Box): action space.
    """

    def __init__(self, obs_space, act_space):
        super(MujocoModel, self).__init__()

        self.fc1 = nn.Linear(obs_space.shape[0], 64)
        self.fc2 = nn.Linear(64, 64)

        self.fc_value = nn.Linear(64, 1)

        self.fc_policy = nn.Linear(64, np.prod(act_space.shape))
        self.fc_pi_std = paddle.static.create_parameter(
            [1, np.prod(act_space.shape)],
            dtype='float32',
            default_initializer=nn.initializer.Constant(value=0))

    def value(self, obs):
        """ Get value network prediction
        Args:
            obs (np.array): current observation
        """
        out = paddle.tanh(self.fc1(obs))
        out = paddle.tanh(self.fc2(out))
        value = self.fc_value(out)
        return value

    def policy(self, obs):
        """ Get policy network prediction
        Args:
            obs (np.array): current observation
        """
        out = paddle.tanh(self.fc1(obs))
        out = paddle.tanh(self.fc2(out))
        action_mean = self.fc_policy(out)

        action_logstd = self.fc_pi_std
        action_std = paddle.exp(action_logstd)
        return action_mean, action_std
