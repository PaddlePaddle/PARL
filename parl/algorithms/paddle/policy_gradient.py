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

import parl
import paddle
from paddle.distribution import Categorical
from parl.utils.utils import check_model_method

__all__ = ['PolicyGradient']


class PolicyGradient(parl.Algorithm):
    def __init__(self, model, lr):
        """Policy gradient algorithm

        Args:
            model (parl.Model): model defining forward network of policy.
            lr (float): learning rate.

        """
        # checks
        check_model_method(model, 'forward', self.__class__.__name__)
        assert isinstance(lr, float)

        self.model = model
        self.optimizer = paddle.optimizer.Adam(
            learning_rate=lr, parameters=self.model.parameters())

    def predict(self, obs):
        """Predict the probability of actions

        Args:
            obs (paddle tensor): shape of (obs_dim,)

        Returns:
            prob (paddle tensor): shape of (action_dim,)
        """
        prob = self.model(obs)
        return prob

    def learn(self, obs, action, reward):
        """Update model with policy gradient algorithm

        Args:
            obs (paddle tensor): shape of (batch_size, obs_dim)
            action (paddle tensor): shape of (batch_size, 1)
            reward (paddle tensor): shape of (batch_size, 1)

        Returns:
            loss (paddle tensor): shape of (1)

        """
        prob = self.model(obs)
        log_prob = Categorical(prob).log_prob(action)
        loss = paddle.mean(-1 * log_prob * reward)

        self.optimizer.clear_grad()
        loss.backward()
        self.optimizer.step()
        return loss
