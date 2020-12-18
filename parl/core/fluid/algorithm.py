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

import parl
from parl.core.algorithm_base import AlgorithmBase

__all__ = ['Algorithm']


class Algorithm(AlgorithmBase):
    """
    | `alias`: ``parl.Algorithm``
    | `alias`: ``parl.core.fluid.algorithm.Algorithm``

    | ``Algorithm`` defines the way how to update the parameters of the ``Model``. This is where we define loss functions and the optimizer of the neural network. An ``Algorithm`` has at least a model.

    | PARL has implemented various algorithms(DQN/DDPG/PPO/A3C/IMPALA) that can be reused quickly, which can be accessed with ``parl.algorithms``.

    Example:

    .. code-block:: python

        import parl

        model = Model()
        dqn = parl.algorithms.DQN(model, lr=1e-3)

    Attributes:
        model(``parl.Model``): a neural network that represents a policy or a Q-value function.

    Pulic Functions:
        - ``get_weights``: return a Python dictionary containing parameters of the current model.
        - ``set_weights``: copy parameters from ``get_weights()`` to the model.
        - ``sample``: return a noisy action to perform exploration according to the policy.
        - ``predict``: return an action given current observation.
        - ``learn``: define the loss function and create an optimizer to minized the loss.

    Note:

        ``Algorithm`` defines all its computation inside a ``fluid.Program``, such that the returns of functions(`sample`, `predict`, `learn`) are tensors.
        ``Agent`` also has functions like `sample`, `predict`, and `learn`, but they return numpy array for the agent.
        
    """

    def __init__(self, model=None):
        """
        Args:
            model(``parl.Model``): a neural network that represents a policy or a Q-value function.
        """
        assert isinstance(model, parl.Model)
        self.model = model

    def get_weights(self, model_ids=None):
        """Get weights of all `parl.Model` in self.__dict__.
        
        If `model_ids` is not None, will only return weights of
        models whose model_id are in `model_ids`.
        
        Note:
            `parl.Model` in list, tuple and dict will be included. But `parl.Model` in
            nested list, tuple and dict won't be included.

        Args:
            model_ids (List/Set): list/set of model_id, will only return weights of models
                              whose model_id in the `model_ids`.

        Returns:
            Dict of weights ({attribute name: numpy array/List/Dict})
        """
        if model_ids is not None:
            assert isinstance(model_ids, (list, set))
            model_ids = set(model_ids)

        model_weights = {}
        for key in self.__dict__.keys():
            value = getattr(self, key)
            if isinstance(value, parl.Model):
                if model_ids is None or value.model_id in model_ids:
                    model_weights[key] = value.get_weights()
            elif isinstance(value, list) or isinstance(value, tuple):
                weights_list = []
                for x in value:
                    if isinstance(x, parl.Model):
                        if model_ids is None or x.model_id in model_ids:
                            weights_list.append(x.get_weights())
                if weights_list:
                    model_weights[key] = weights_list
            elif isinstance(value, dict):
                weights_dict = {}
                for sub_k, sub_v in value.items():
                    if isinstance(sub_v, parl.Model):
                        if model_ids is None or sub_v.model_id in model_ids:
                            weights_dict[sub_k] = sub_v.get_weights()
                if weights_dict:
                    model_weights[key] = weights_dict
        return model_weights

    def set_weights(self, weights, model_ids=None):
        """Set weights of all `parl.Model` in self.__dict__.

        If `model_ids` is not None, will only set weights of
        models whose model_id are in `model_ids`.

        Note:
            `parl.Model` in list, tuple and dict will be included. But `parl.Model` in
            nested list, tuple and dict won't be included.

        Args:
            weights (Dict): Dict of weights ({attribute name: numpy array/List/Dict})
            model_ids (List/Set): list/set of model_id, will only set weights of models
                              whiose model_id in the `model_ids`.
        
        """
        assert isinstance(weights, dict)
        if model_ids is not None:
            assert isinstance(model_ids, (list, set))
            model_ids = set(model_ids)

        for key in self.__dict__.keys():
            value = getattr(self, key)
            if isinstance(value, parl.Model):
                if model_ids is None or value.model_id in model_ids:
                    assert key in weights, "weights is inconsistent with current algorithm and given model_ids."
                    value.set_weights(weights[key])
            elif isinstance(value, list) or isinstance(value, tuple):
                model_list = []
                for x in value:
                    if isinstance(x, parl.Model):
                        if model_ids is None or x.model_id in model_ids:
                            model_list.append(x)
                if model_list:
                    assert key in weights and len(model_list) == len(weights[key]), \
                            "weights is inconsistent with current algorithm and given model_ids."
                    for i, model in enumerate(model_list):
                        model.set_weights(weights[key][i])
            elif isinstance(value, dict):
                model_dict = {}
                for sub_k, sub_v in value.items():
                    if isinstance(sub_v, parl.Model):
                        if model_ids is None or sub_v.model_id in model_ids:
                            model_dict[sub_k] = sub_v
                if model_dict:
                    assert key in weights and set(model_dict.keys()) == set(weights[key].keys()), \
                            "weights is inconsistent with current algorithm and given model_ids."
                    for sub_k, model in model_dict.items():
                        model.set_weights(weights[key][sub_k])

    def get_model_ids(self):
        """Get model_id of all `parl.Model` in self.__dict__.

        Note:
            `parl.Model` in list, tuple and dict will be included. But `parl.Model` in
            nested list, tuple and dict won't be included.
        
        Returns:
            Set of model_id 
        """
        model_ids = set([])
        for key in self.__dict__.keys():
            value = getattr(self, key)
            if isinstance(value, parl.Model):
                model_ids.add(value.model_id)
            elif isinstance(value, list) or isinstance(value, tuple):
                for x in value:
                    if isinstance(x, parl.Model):
                        model_ids.add(x.model_id)
            elif isinstance(value, dict):
                for sub_k, sub_v in value.items():
                    if isinstance(sub_v, parl.Model):
                        model_ids.add(sub_v.model_id)
        return model_ids

    @property
    def model_ids(self):
        return self.get_model_ids()

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
