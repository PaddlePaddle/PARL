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

from abc import ABCMeta, abstractmethod
from parl.framework.model_base import Model

__all__ = ['Algorithm']


class Algorithm(object):
    """
    Algorithm defines the way how we update the model. For example,
    after defining forward network in `Model` class, you should define how to update the model here.
    Before creating a customized algorithm, please do check algorithms of PARL.
    Most common used algorithms like DQN/DDPG/PPO/A3C have been providing in algorithms, go and have a try.
    It's easy to use them and just try parl.algorithms.DQN.

    An Algorithm implements two functions:
    1. define_predict() build forward process which was defined in `Model`
    2. define_learn() computes a cost for optimization

    An algorithm should be updating part of a network. The user only needs to 
    implement the rest of the network(forward) in the Model class.
    """

    def __init__(self, model, hyperparas=None):
        assert isinstance(model, Model)
        self.model = model
        self.hp = hyperparas

    def define_predict(self, obs):
        """
        describe process for building predcition program
        """
        raise NotImplementedError()

    def define_learn(self, obs, action, reward, next_obs, terminal):
        """define how to update the model here, you may need to do the following:
            1. define a cost for optimization
            2. specify your optimizer
            3. optimize model defined in Model
        """
        raise NotImplementedError()

    def get_params(self):
        """ Get parameters of self.model

        Returns:
            List of numpy array. 
        """
        return self.model.get_params()

    def set_params(self, params, gpu_id):
        """ Set parameters of self.model

        Args:
            params: List of numpy array.
            gpu_id: gpu id where self.model in. (if gpu_id < 0, means in cpu.)
        """
        self.model.set_params(params, gpu_id=gpu_id)
