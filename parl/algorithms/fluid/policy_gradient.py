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
from parl.utils.deprecation import deprecated

__all__ = ['PolicyGradient']


class PolicyGradient(Algorithm):
    def __init__(self, model, hyperparas=None, lr=None):
        """ Policy Gradient algorithm
        
        Args:
            model (parl.Model): forward network of the policy.
            hyperparas (dict): (deprecated) dict of hyper parameters.
            lr (float): learning rate of the policy model.
        """

        self.model = model
        if hyperparas is not None:
            warnings.warn(
                "the `hyperparas` argument of `__init__` function in `parl.Algorithms.PolicyGradient` is deprecated since version 1.2 and will be removed in version 1.3.",
                DeprecationWarning,
                stacklevel=2)
            self.lr = hyperparas['lr']
        else:
            assert isinstance(lr, float)
            self.lr = lr

    @deprecated(
        deprecated_in='1.2', removed_in='1.3', replace_function='predict')
    def define_predict(self, obs):
        """ use policy model self.model to predict the action probability
        """
        return self.predict(obs)

    def predict(self, obs):
        """ use policy model self.model to predict the action probability
        """
        return self.model(obs)

    @deprecated(
        deprecated_in='1.2', removed_in='1.3', replace_function='learn')
    def define_learn(self, obs, action, reward):
        """ update policy model self.model with policy gradient algorithm
        """
        return self.learn(obs, action, reward)

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
