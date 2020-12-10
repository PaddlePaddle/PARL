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

from parl.core.model_base import ModelBase


class AlgorithmBase(object):
    """`AlgorithmBase` is the base class of the `parl.Algorithm` in different
    frameworks.

    This base class mainly do the following things:
        1. Implements APIs to set or get weights of all `ModelBase` in self.__dict__;
        2. Defines common APIs that `parl.Algorithm` should implement in different frameworks.
    """

    def __init__(self):
        pass

    def get_weights(self):
        """Get weights of all `ModelBase` in self.__dict__.
        
        Note:
            `ModelBase` in list, tuple and dict will be included. But `ModelBase` in
            nested list, tuple and dict won't be included.

        Returns:
            Dict of weights ({attribute name: numpy array/List/Dict})
        """
        model_weights = {}
        for key in self.__dict__.keys():
            value = getattr(self, key)
            if isinstance(value, ModelBase):
                model_weights[key] = value.get_weights()
            elif isinstance(value, list) or isinstance(value, tuple):
                weights_list = []
                for x in value:
                    if isinstance(x, ModelBase):
                        weights_list.append(x.get_weights())
                if weights_list:
                    model_weights[key] = weights_list
            elif isinstance(value, dict):
                weights_dict = {}
                for sub_k, sub_v in value.items():
                    if isinstance(sub_v, ModelBase):
                        weights_dict[sub_k] = sub_v.get_weights()
                if weights_dict:
                    model_weights[key] = weights_dict
        return model_weights

    def set_weights(self, weights):
        """Set weights of all `ModelBase` in self.__dict__.

        Note:
            `ModelBase` in list, tuple and dict will be included. But `ModelBase` in
            nested list, tuple and dict won't be included.

        Args:
            weights (Dict): Dict of weights ({attribute name: numpy array/List/Dict})
        
        """
        for key in self.__dict__.keys():
            value = getattr(self, key)
            if isinstance(value, ModelBase):
                assert key in weights, "weights is inconsistent with current algorithm."
                value.set_weights(weights[key])
            elif isinstance(value, list) or isinstance(value, tuple):
                model_list = []
                for x in value:
                    if isinstance(x, ModelBase):
                        model_list.append(x)
                if model_list:
                    assert key in weights and len(model_list) == len(weights[key]), \
                        "weights is inconsistent with current algorithm."
                    for i, model in enumerate(model_list):
                        model.set_weights(weights[key][i])
            elif isinstance(value, dict):
                model_dict = {}
                for sub_k, sub_v in value.items():
                    if isinstance(sub_v, ModelBase):
                        model_dict[sub_k] = sub_v
                if model_dict:
                    assert key in weights and set(model_dict.keys()) == set(weights[key].keys()), \
                        "weights is inconsistent with current algorithm."
                    for sub_k, model in model_dict.items():
                        model.set_weights(weights[key][sub_k])

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
