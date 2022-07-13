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

from parl.core.algorithm_base import AlgorithmBase
from parl.core.paddle.model import Model

__all__ = ['Algorithm']


class Algorithm(AlgorithmBase):
    """
    | `alias`: ``parl.Algorithm``
    | `alias`: ``parl.core.fluid.algorithm.Algorithm``

    | ``Algorithm`` defines the way how to update the parameters of 
    the ``Model``. This is where we define loss functions and the 
    optimizer of the neural network. An ``Algorithm`` has at least 
    a model.

    | PARL has implemented various algorithms(DQN/DDPG/PPO/A3C/IMPALA) that 
    can be reused quickly, which can be accessed with ``parl.algorithms``.

    Example:

    .. code-block:: python

        import parl

        model = Model()
        dqn = parl.algorithms.DQN(model, lr=1e-3)

    Attributes:
        model(``parl.Model``): a neural network that represents a policy 
        or a Q-value function.

    Pulic Functions:
        - ``get_weights``: return a Python dictionary containing parameters 
        of the current model.
        - ``set_weights``: copy parameters from ``get_weights()`` to the model.
        - ``sample``: return a noisy action to perform exploration according 
        to the policy.
        - ``predict``: return an action given current observation.
        - ``learn``: define the loss function and create an optimizer to 
        minized the loss.
    """

    def __init__(self, model=None):
        """
        Args:
            model(``parl.Model``): a neural network that represents a policy or a Q-value function.
        """
        assert isinstance(model, Model)
        self.model = model

    def learn(self, *args, **kwargs):
        """ Define the loss function and create an optimizer to minize the loss.
        """
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        """ Refine the predicting process, e.g,. use the policy model to predict actions.
        """
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        """ Define the sampling process. This function returns an action with noise to perform exploration.
        """
        raise NotImplementedError

    def get_weights(self):
        """ Get weights of self.model.

        Returns:
            weights (dict): a Python dict containing the parameters of self.model.
        """
        return self.model.get_weights()

    def set_weights(self, params):
        """ Set weights from ``get_weights`` to the model.

        Args:
            weights (dict): a Python dict containing the parameters of self.model.
        """
        self.model.set_weights(params)
