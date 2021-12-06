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
import parl
import paddle
from parl.utils.utils import check_model_method

__all__ = ['DQN']


class DQN(parl.Algorithm):
    def __init__(self, model, gamma=None, lr=None):
        """ DQN algorithm

        Args:
            model (parl.Model): forward neural network representing the Q function.
            gamma (float): discounted factor for `accumulative` reward computation
            lr (float): learning rate.
        """
        # checks
        check_model_method(model, 'forward', self.__class__.__name__)
        assert isinstance(gamma, float)
        assert isinstance(lr, float)

        self.model = model
        self.target_model = copy.deepcopy(model)

        self.gamma = gamma
        self.lr = lr

        self.mse_loss = paddle.nn.MSELoss(reduction='mean')
        self.optimizer = paddle.optimizer.Adam(
            learning_rate=lr, parameters=self.model.parameters())

    def predict(self, obs):
        """ use self.model (Q function) to predict the action values
        """
        return self.model(obs)

    def learn(self, obs, action, reward, next_obs, terminal):
        """ update the Q function (self.model) with DQN algorithm
        """
        # Q
        pred_values = self.model(obs)
        action_dim = pred_values.shape[-1]
        action = paddle.squeeze(action, axis=-1)
        action_onehot = paddle.nn.functional.one_hot(
            action, num_classes=action_dim)
        pred_value = pred_values * action_onehot
        pred_value = paddle.sum(pred_value, axis=1, keepdim=True)

        # target Q
        with paddle.no_grad():
            max_v = self.target_model(next_obs).max(1, keepdim=True)
            target = reward + (1 - terminal) * self.gamma * max_v
        loss = self.mse_loss(pred_value, target)

        # optimize
        self.optimizer.clear_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def sync_target(self):
        """ assign the parameters of the training network to the target network
        """
        self.model.sync_weights_to(self.target_model)
