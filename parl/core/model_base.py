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


class ModelBase(object):
    """`ModelBase` is the base class of the `parl.Model` in different frameworks.

    This base class mainly do the following things:
        1. Implements APIs to manage model_id of the `parl.Model`; 
        2. Defines common APIs that `parl.Model` should implement in different frameworks.
    """

    def __init__(self):
        pass

    def forward(self, *args, **kwargs):
        """Define forward network of the model.
        """
        raise NotImplementedError

    def get_weights(self, *args, **kwargs):
        """Get weights of the model.
        """
        raise NotImplementedError

    def set_weights(self, weights, *args, **kwargs):
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
