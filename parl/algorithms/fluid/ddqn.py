#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import warnings
warnings.simplefilter('default')

import copy
import numpy as np
import paddle.fluid as fluid
from parl.core.fluid.algorithm import Algorithm
from parl.core.fluid import layers


class DDQN(Algorithm):
    def __init__(
            self,
            model,
            act_dim=None,
            gamma=None,
    ):
        """ Double DQN algorithm

        Args:
            model (parl.Model): model defining forward network of Q function.
            gamma (float): discounted factor for reward computation.
        """
        self.model = model
        self.target_model = copy.deepcopy(model)

        assert isinstance(act_dim, int)
        assert isinstance(gamma, float)

        self.act_dim = act_dim
        self.gamma = gamma

    def predict(self, obs):
        return self.model.value(obs)

    def learn(self, obs, action, reward, next_obs, terminal, learning_rate):
        pred_value = self.model.value(obs)
        action_onehot = layers.one_hot(action, self.act_dim)
        action_onehot = layers.cast(action_onehot, dtype='float32')
        pred_action_value = layers.reduce_sum(
            layers.elementwise_mul(action_onehot, pred_value), dim=1)

        # choose acc. to behavior network
        next_action_value = self.model.value(next_obs)
        greedy_action = layers.argmax(next_action_value, axis=-1)

        # calculate the target q value with target network
        batch_size = layers.cast(layers.shape(greedy_action)[0], dtype='int')
        range_tmp = layers.range(
            start=0, end=batch_size, step=1, dtype='int64') * self.act_dim
        a_indices = range_tmp + greedy_action
        a_indices = layers.cast(a_indices, dtype='int32')
        next_pred_value = self.target_model.value(next_obs)
        next_pred_value = layers.reshape(
            next_pred_value, shape=[
                -1,
            ])
        max_v = layers.gather(next_pred_value, a_indices)
        max_v = layers.reshape(
            max_v, shape=[
                -1,
            ])
        max_v.stop_gradient = True

        target = reward + (
            1.0 - layers.cast(terminal, dtype='float32')) * self.gamma * max_v
        cost = layers.square_error_cost(pred_action_value, target)
        cost = layers.reduce_mean(cost)
        optimizer = fluid.optimizer.Adam(
            learning_rate=learning_rate, epsilon=1e-3)
        optimizer.minimize(cost)
        return cost

    def sync_target(self, gpu_id=None):
        """ sync weights of self.model to self.target_model
        """
        if gpu_id is not None:
            warnings.warn(
                "the `gpu_id` argument of `sync_target` function in `parl.Algorithms.DQN` is deprecated since version 1.2 and will be removed in version 1.3.",
                DeprecationWarning,
                stacklevel=2)
        self.model.sync_weights_to(self.target_model)
