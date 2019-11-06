#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import torch
import torch.nn as nn

from parl.core.model_base import ModelBase
from parl.utils import machine_info

__all__ = ['Model']


class Model(nn.Module, ModelBase):
    """
    | `alias`: ``parl.Model``
    | `alias`: ``parl.core.torch.agent.Model``

    | ``Model`` is a base class of PARL for the neural network. A ``Model`` is
    usually a policy or Q-value function, which predicts an action or an
    estimate according to the environmental observation.

    | To use the ``PyTorch`` backend model, user needs to call ``super(Model,
    self).__init__()`` at the beginning of ``__init__`` function.

    | ``Model`` supports duplicating a ``Model`` instance in a pythonic way:

    | ``copied_model = copy.deepcopy(model)``

    Example:

    .. code-block:: python

        import parl
        import torch.nn as nn

        class Policy(parl.Model):
            def __init__(self):
                super(Policy, self).__init__()
                self.fc = nn.Linear(in_features=100, out_features=32)

            def policy(self, obs):
                out = self.fc(obs)
                return out

        policy = Policy()
        copied_policy = copy.deepcopy(model)

    Attributes:
        model_id(str): each model instance has its unique model_id.

    Public Functions:
        - ``sync_weights_to``: synchronize parameters of the current model to
        another model.
        - ``get_weights``: return a list containing all the parameters of the
        current model.
        - ``set_weights``: copy parameters from ``set_weights()`` to the model.
        - ``forward``: define the computations of a neural network. **Should**
        be overridden by all subclasses.

    """

    def __init___(self):
        super(Model, self).__init__()

    def sync_weights_to(self, target_model, decay=0.0):
        """Synchronize parameters of current model to another model.

        target_model_weights = decay * target_model_weights + (1 - decay) *
        current_model_weights

        Args:
            target_model (`parl.Model`): an instance of ``Model`` that has the
            same neural network architecture as the current model.
            decay (float):  the rate of decline in copying parameters. 0 if no
            parameters decay when synchronizing the parameters.

        Example:

        .. code-block:: python

            import copy
            # create a model that has the same neural network structures.
            target_model = copy.deepcopy(model)

            # after initializing the parameters ...
            model.sync_weights_to(target_model)

        Note:
            Before calling ``sync_weights_to``, parameters of the model must
            have been initialized.
        """

        assert not target_model is self, "cannot copy between identical model"
        assert isinstance(target_model, Model)
        assert self.__class__.__name__ == target_model.__class__.__name__, \
            "must be the same class for params syncing!"
        assert (decay >= 0 and decay <= 1)

        target_vars = dict(target_model.named_parameters())
        for name, var in self.named_parameters():
            target_vars[name].data.copy_(decay * target_vars[name].data +
                                         (1 - decay) * var.data)

    def get_weights(self):
        """Returns a Python list containing parameters of current model.

        Returns: a Python list containing the parameters of current model.
        """
        weights = self.state_dict()
        for key in weights.keys():
            weights[key] = weights[key].cpu().numpy()
        return weights

    def set_weights(self, weights):
        """Copy parameters from ``set_weights()`` to the model.
        
        Args:
            weights (list): a Python list containing the parameters.
        """
        for key in weights.keys():
            weights[key] = torch.from_numpy(weights[key])
        self.load_state_dict(weights)
