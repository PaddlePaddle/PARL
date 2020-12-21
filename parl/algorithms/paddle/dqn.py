#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import numpy as np
import parl
import paddle
import paddle.fluid as fluid
from cartpole_model import CartpoleModel

__all__ = ['DQN']


class DQN(parl.Algorithm):
    def __init__(self, model, gamma=None, lr=None):
        """ DQN algorithm

        Args:
            model (parl.Model): model defining forward network of Q function.
            gamma (float): discounted factor for reward computation.
            lr (float): learning rate.
        """
        self.model = model
        self.target_model = copy.deepcopy(model)

        assert isinstance(gamma, float)
        assert isinstance(lr, float)
        self.gamma = gamma
        self.lr = lr

        self.mse_loss = paddle.nn.MSELoss(reduction='mean')
        self.optimizer = paddle.optimizer.Adam(
            learning_rate=lr, parameters=self.model.parameters())

    def predict(self, obs):
        """ use value model self.model to predict the action value
        """
        return self.model.value(obs)

    def learn(self, obs, action, reward, next_obs, terminal):
        """ update value model self.model with DQN algorithm
        """
        # Q
        pred_values = self.model.value(obs)
        action_dim = pred_values.shape[-1]
        action_onehot = paddle.fluid.layers.one_hot(input=action, depth=action_dim)
        pred_value = paddle.multiply(pred_values, action_onehot)
        pred_value = paddle.sum(pred_value, axis=1, keepdim=True)

        # target Q
        with fluid.dygraph.no_grad():
            max_v = self.target_model.value(next_obs).max(1, keepdim=True)
            target = reward + (1 - terminal) * self.gamma * max_v
        loss = self.mse_loss(pred_value, target)

        # optimize
        loss.backward()
        self.optimizer.step()
        self.optimizer.clear_grad()
        return loss

    def sync_target(self):
        self.model.sync_weights_to(self.target_model)
