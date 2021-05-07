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
import parl
from parl import layers


class MAModel(parl.Model):
    def __init__(self, act_dim):
        self.actor_model = ActorModel(act_dim)
        self.critic_model = CriticModel()

    def policy(self, obs):
        return self.actor_model.policy(obs)

    def value(self, obs, act):
        return self.critic_model.value(obs, act)

    def get_actor_params(self):
        return self.actor_model.parameters()

    def get_critic_params(self):
        return self.critic_model.parameters()


class ActorModel(parl.Model):
    def __init__(self, act_dim):
        hid1_size = 64
        hid2_size = 64

        self.fc1 = layers.fc(
            size=hid1_size,
            act='relu',
            param_attr=fluid.initializer.Normal(loc=0.0, scale=0.1))
        self.fc2 = layers.fc(
            size=hid2_size,
            act='relu',
            param_attr=fluid.initializer.Normal(loc=0.0, scale=0.1))
        self.fc3 = layers.fc(
            size=act_dim,
            act=None,
            param_attr=fluid.initializer.Normal(loc=0.0, scale=0.1))

    def policy(self, obs):
        hid1 = self.fc1(obs)
        hid2 = self.fc2(hid1)
        means = self.fc3(hid2)
        means = means
        return means


class CriticModel(parl.Model):
    def __init__(self):
        hid1_size = 64
        hid2_size = 64

        self.fc1 = layers.fc(
            size=hid1_size,
            act='relu',
            param_attr=fluid.initializer.Normal(loc=0.0, scale=0.1))
        self.fc2 = layers.fc(
            size=hid2_size,
            act='relu',
            param_attr=fluid.initializer.Normal(loc=0.0, scale=0.1))
        self.fc3 = layers.fc(
            size=1,
            act=None,
            param_attr=fluid.initializer.Normal(loc=0.0, scale=0.1))

    def value(self, obs_n, act_n):
        inputs = layers.concat(obs_n + act_n, axis=1)
        hid1 = self.fc1(inputs)
        hid2 = self.fc2(hid1)
        Q = self.fc3(hid2)
        Q = layers.squeeze(Q, axes=[1])
        return Q
