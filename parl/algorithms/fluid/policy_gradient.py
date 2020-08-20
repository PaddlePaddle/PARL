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

import warnings
warnings.simplefilter('default')

import paddle.fluid as fluid
from parl.core.fluid.algorithm import Algorithm
from parl.core.fluid import layers

__all__ = ['PolicyGradient']


class PolicyGradient(Algorithm):
    def __init__(self, model, lr=None):
        """ Policy Gradient algorithm
        
        Args:
            model (parl.Model): forward network of the policy.
            lr (float): learning rate of the policy model.
        """

        self.model = model
        assert isinstance(lr, float)
        self.lr = lr

    def predict(self, obs):
        """ use policy model self.model to predict the action probability
        """
        return self.model(obs)

    def learn(self, obs, action, reward):
        """ update policy model self.model with policy gradient algorithm
        """
        act_prob = self.model(obs)
        log_prob = layers.cross_entropy(act_prob, action)
        cost = log_prob * reward
        cost = layers.reduce_mean(cost)
        optimizer = fluid.optimizer.Adam(self.lr)
        optimizer.minimize(cost)
        return cost
