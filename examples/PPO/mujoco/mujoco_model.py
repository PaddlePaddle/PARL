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
        obs_dim (int): observation dimension.
        act_dim (int): action dimension.
    """

    def __init__(self, obs_dim, act_dim, init_logvar=0.0):
        super(MujocoModel, self).__init__()
        super(MujocoModel, self).__init__()
        self.policy_model = PolicyModel(obs_dim, act_dim, init_logvar)
        self.value_model = ValueModel(obs_dim)
        self.policy_lr = self.policy_model.lr
        self.value_lr = self.value_model.lr

    def value(self, obs):
        """ Get value network prediction
        Args:
            obs (np.array): current observation
        """
        return self.value_model.value(obs)

    def policy(self, obs):
        """ Get policy network prediction
        Args:
            obs (np.array): current observation
        """
        return self.policy_model.policy(obs)


class PolicyModel(parl.Model):
    def __init__(self, obs_dim, act_dim, init_logvar):
        super(PolicyModel, self).__init__()
        self.policy_logvar = -1.0
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        hid1_size = obs_dim * 10
        hid3_size = act_dim * 10
        hid2_size = int(np.sqrt(hid1_size * hid3_size))

        self.lr = 9e-4 / np.sqrt(hid2_size)

        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.fc2 = nn.Linear(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, hid3_size)
        self.fc_policy = nn.Linear(hid3_size, act_dim)

        # logvar_speed is used to 'fool' gradient descent into making faster updates to log-variances.
        # heuristic sets logvar_speed based on network size.
        logvar_speed = (10 * hid3_size) // 48  # default setting
        # logvar_speed = (10 * hid3_size) // 8  # finetuned for Humanoid-v2 to achieve fast convergence
        self.fc_pi_std = paddle.create_parameter([logvar_speed, act_dim],
                                                 dtype='float32',
                                                 default_initializer=nn.initializer.Constant(value=init_logvar))

    def policy(self, obs):
        hid1 = paddle.tanh(self.fc1(obs))
        hid2 = paddle.tanh(self.fc2(hid1))
        hid3 = self.fc3(hid2)
        means = self.fc_policy(hid3)
        logvars = paddle.sum(self.fc_pi_std, axis=0) + self.policy_logvar
        return means, logvars


class ValueModel(parl.Model):
    def __init__(self, obs_dim):
        super(ValueModel, self).__init__()
        hid1_size = obs_dim * 10
        hid3_size = 5
        hid2_size = int(np.sqrt(hid1_size * hid3_size))

        self.lr = 1e-2 / np.sqrt(hid2_size)  # 1e-3 empirically determined

        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.fc2 = nn.Linear(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, hid3_size)
        self.fc_value = nn.Linear(hid3_size, 1)

    def value(self, obs):
        hid1 = paddle.tanh(self.fc1(obs))
        hid2 = paddle.tanh(self.fc2(hid1))
        hid3 = paddle.tanh(self.fc3(hid2))
        value = self.fc_value(hid3)
        return value
