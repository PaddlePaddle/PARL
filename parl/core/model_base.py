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

from parl.core.model_helper import global_model_helper


class ModelBase(object):
    """`ModelBase` is the base class of the `parl.Model` in different frameworks.

    This base class mainly do the following things:
        1. Implements APIs to manage model_id of the `parl.Model`; 
        2. Defines common APIs that `parl.Model` should implement in different frameworks.
    """

    def __init__(self, model_id=None):
        """
        Args:
            model_id (String): user-specified model_id (default: None)
        """
        if model_id is not None:
            global_model_helper.register_model_id(model_id)
            self.__model_id = model_id
        else:
            self.__model_id = global_model_helper.generate_model_id()

    @property
    def model_id(self):
        return self.get_model_id()

    @model_id.setter
    def model_id(self, model_id):
        self.set_model_id(model_id)

    def get_model_id(self):
        """Get model_id of `ModelBase`.
        If not created, will create a new model_id.

        Returns:
            String of model_id.
        """
        try:
            return self.__model_id
        except AttributeError:
            self.__model_id = global_model_helper.generate_model_id()
            return self.__model_id

    def set_model_id(self, model_id):
        """Set model_id of `ModelBase` with given model_id.
        
        Args:
            model_id (string): string of model_id.
        """
        global_model_helper.register_model_id(model_id)
        self.__model_id = model_id

    def forward(self, *args, **kwargs):
        """Define forward network of the model.
        """
        raise NotImplementedError

    def get_weights(self):
        """Get weights of the model.
        """
        raise NotImplementedError

    def set_weights(self, weights):
        """Set weights of the model with given weights.
        """
        raise NotImplementedError

    def sync_weights_to(self, other_model):
        """Synchronize weights of the model to another model.
        """
        raise NotImplementedError

    def parameters(self):
        """Get the parameters of the model.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """Call forward function.
        """
        return self.forward(*args, **kwargs)
