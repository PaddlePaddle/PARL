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
"""
Base class to define an Algorithm.
"""

from abc import ABCMeta, abstractmethod
from parl.utils.utils import has_fun

__all__ = ['Network', 'Model']


class Network(object):
    """
    A Network is an unordered set of LayerFuncs or Networks.
    """

    def sync_paras_to(self, target_net):
        assert not target_net is self, "cannot copy between identical networks"
        assert isinstance(target_net, Network)
        assert self.__class__.__name__ == target_net.__class__.__name__, \
            "must be the same class for para syncing!"

        for attr in self.__dict__:
            if not attr in target_net.__dict__:
                continue
            val = getattr(self, attr)
            target_val = getattr(target_net, attr)

            assert type(val) == type(target_val)
            ### TODO: sync paras recursively
            if has_fun(val, 'sync_paras_to'):
                val.sync_paras_to(target_val)
            elif isinstance(val, tuple) or isinstance(val, list) or isinstance(
                    val, set):
                for v, tv in zip(val, target_val):
                    v.sync_paras_to(tv)
            elif isinstance(val, dict):
                for k in val.keys():
                    assert k in target_val
                    val[k].sync_paras_to(target_val[k])
            else:
                # for any other type, we do not copy
                pass


class Model(Network):
    """
    A Model is owned by an Algorithm. 
    It implements the entire network model(forward part) to solve a specific problem.
    In conclusion, Model is responsible for forward and 
    Algorithm is responsible for backward.

    Model can also be used to construct target model, which has the same structure as initial model.
    Here is an example:
        ```python
        class Actor(Model):
            __init__(self, obs_dim, act_dim):
                self.obs_dim = obs_dim
                self.act_dim = act_dim
                self.fc1 = layers.fc(size=128, act='relu')
                self.fc2 = layers.fc(size=64, act='relu')
        actor = Actor(obs_dim=12, act_dim=2)
        target_actor = copy.deepcopy(actor)
        ```

    Note that it's the model structure that is copied from initial actor,
    parameters in initial model havn't been copied to target model.
    To copy parameters, you must explicitly use sync_paras_to function after the program is initialized.

    """
    __metaclass__ = ABCMeta

    def __init__(self):
        super(Model, self).__init__()

    def policy(self, *args):
        """
        Implement your policy here. 
        The function was later used by algorithm 
        Return: action_dists: a dict of action distribution objects
                states
        Optional: a model might not always have to implement policy()
        """
        raise NotImplementedError()

    def value(self, *args):
        """
        Return: values: a dict of estimated values for the current observations and states
                        For example, "q_value" and "v_value"
        Optional: a model might not always have to implement value()
        """
        raise NotImplementedError()
