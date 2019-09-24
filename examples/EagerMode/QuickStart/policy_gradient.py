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
import paddle.fluid.layers as layers


class PolicyGradient(object):
    def __init__(self, model, lr):
        self.model = model
        self.optimizer = fluid.optimizer.Adam(learning_rate=lr)

    def predict(self, obs):
        obs = fluid.dygraph.to_variable(obs)
        obs = layers.cast(obs, dtype='float32')
        return self.model(obs)

    def learn(self, obs, action, reward):
        obs = fluid.dygraph.to_variable(obs)
        obs = layers.cast(obs, dtype='float32')
        act_prob = self.model(obs)
        action = fluid.dygraph.to_variable(action)
        reward = fluid.dygraph.to_variable(reward)

        log_prob = layers.cross_entropy(act_prob, action)
        cost = log_prob * reward
        cost = layers.cast(cost, dtype='float32')
        cost = layers.reduce_mean(cost)
        cost.backward()
        self.optimizer.minimize(cost)
        self.model.clear_gradients()
        return cost
