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
        """Get weights of all `ModelBase`
        """
        raise NotImplementedError

    def set_weights(self, weights):
        """Set weights of all `ModelBase`
        """
        raise NotImplementedError

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
