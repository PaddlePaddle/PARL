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
import parl
from parl import layers
from paddle import fluid
from paddle.fluid.param_attr import ParamAttr


class MujocoModel(parl.Model):
    def __init__(self, obs_dim, act_dim, init_logvar=-1.0):
        self.policy_model = PolicyModel(obs_dim, act_dim, init_logvar)
        self.value_model = ValueModel(obs_dim, act_dim)
        self.policy_lr = self.policy_model.lr
        self.value_lr = self.value_model.lr

    def policy(self, obs):
        return self.policy_model.policy(obs)

    def policy_sample(self, obs):
        return self.policy_model.sample(obs)

    def value(self, obs):
        return self.value_model.value(obs)


class PolicyModel(parl.Model):
    def __init__(self, obs_dim, act_dim, init_logvar):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        hid1_size = obs_dim * 10
        hid3_size = act_dim * 10
        hid2_size = int(np.sqrt(hid1_size * hid3_size))

        self.lr = 9e-4 / np.sqrt(hid2_size)

        self.fc1 = layers.fc(size=hid1_size, act='tanh')
        self.fc2 = layers.fc(size=hid2_size, act='tanh')
        self.fc3 = layers.fc(size=hid3_size, act='tanh')
        self.fc4 = layers.fc(size=act_dim, act='tanh')

        self.logvars = layers.create_parameter(
            shape=[act_dim],
            dtype='float32',
            default_initializer=fluid.initializer.ConstantInitializer(
                init_logvar))

    def policy(self, obs):
        hid1 = self.fc1(obs)
        hid2 = self.fc2(hid1)
        hid3 = self.fc3(hid2)
        means = self.fc4(hid3)
        logvars = self.logvars()
        return means, logvars

    def sample(self, obs):
        means, logvars = self.policy(obs)
        sampled_act = means + (
            layers.exp(logvars / 2.0) *  # stddev
            layers.gaussian_random(shape=(self.act_dim, ), dtype='float32'))
        return sampled_act


class ValueModel(parl.Model):
    def __init__(self, obs_dim, act_dim):
        super(ValueModel, self).__init__()
        hid1_size = obs_dim * 10
        hid3_size = 5
        hid2_size = int(np.sqrt(hid1_size * hid3_size))

        self.lr = 1e-2 / np.sqrt(hid2_size)

        self.fc1 = layers.fc(size=hid1_size, act='tanh')
        self.fc2 = layers.fc(size=hid2_size, act='tanh')
        self.fc3 = layers.fc(size=hid3_size, act='tanh')
        self.fc4 = layers.fc(size=1)

    def value(self, obs):
        hid1 = self.fc1(obs)
        hid2 = self.fc2(hid1)
        hid3 = self.fc3(hid2)
        V = self.fc4(hid3)
        V = layers.squeeze(V, axes=[])
        return V
