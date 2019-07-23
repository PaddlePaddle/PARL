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

from parl.core.algorithm_base import AlgorithmBase
from parl.core.fluid.model import Model
from parl.utils.deprecation import deprecated

__all__ = ['Algorithm']


class Algorithm(AlgorithmBase):
    """Algorithm defines the way how we update the model.
    
    To implement a new algorithm, you may need implement the learn/predict/sample functions.

    Before creating a customized algorithm, please do check algorithms of PARL.
    Most common used algorithms like DQN/DDPG/PPO/A3C/IMPALA have been provided in `parl.algorithms`,
    go and have a try.
    """

    def __init__(self, model=None, hyperparas=None):
        if model is not None:
            warnings.warn(
                "the `model` argument of `__init__` function in `parl.Algorithm` is deprecated since version 1.2 and will be removed in version 1.3.",
                DeprecationWarning,
                stacklevel=2)

            assert isinstance(model, Model)
            self.model = model
        if hyperparas is not None:
            warnings.warn(
                "the `hyperparas` argument of `__init__` function in `parl.Algorithm` is deprecated since version 1.2 and will be removed in version 1.3.",
                DeprecationWarning,
                stacklevel=2)

            self.hp = hyperparas

    @deprecated(
        deprecated_in='1.2', removed_in='1.3', replace_function='get_weights')
    def get_params(self):
        """ Get parameters of self.model

        Returns:
            List of numpy array. 
        """
        return self.model.get_params()

    @deprecated(
        deprecated_in='1.2', removed_in='1.3', replace_function='set_weights')
    def set_params(self, params):
        """ Set parameters of self.model

        Args:
            params: List of numpy array.
        """
        self.model.set_params(params)

    def learn(self, *args, **kwargs):
        """ define learning process, such as how to optimize the model.
        """
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        """ define predicting process, such as using policy model to predict actions when given observations.
        """
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        """ define sampling process, such as using policy model to sample actions when given observations.
        """
        raise NotImplementedError
