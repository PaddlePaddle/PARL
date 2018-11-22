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

import paddle.fluid as fluid
from abc import ABCMeta, abstractmethod
from parl.utils.utils import has_func
from parl.layers.layer_wrappers import LayerFunc

__all__ = ['Network', 'Model']


def _fetch_framework_var(attr_name, is_bias):
    scope = fluid.executor.global_scope()
    core_var = scope.find_var(attr_name)
    shape = core_var.get_tensor().shape()
    framework_var = fluid.layers.create_parameter(
        shape=shape,
        dtype='float32',
        attr=fluid.ParamAttr(name=attr_name),
        is_bias=is_bias)
    return framework_var


def _gen_paras_pair(src, target):
    """ Recursively iterate parameters
    """
    paras_pair = []
    if isinstance(src, Network):
        for attr in src.__dict__:
            if not attr in target.__dict__:
                continue
            src_var = getattr(src, attr)
            target_var = getattr(target, attr)
            paras_pair.extend(_gen_paras_pair(src_var, target_var))
    elif isinstance(src, LayerFunc):
        paras_pair.append((src.param_attr.name, target.param_attr.name, False))
        if src.bias_attr:
            paras_pair.append((src.bias_attr.name, target.bias_attr.name,
                               True))
    elif isinstance(src, tuple) or isinstance(src, list) or isinstance(
            src, set):
        for src_var, target_var in zip(src, target):
            paras_pair.extend(_gen_paras_pair(src_var, target_var))
    elif isinstance(src, dict):
        for k in src.keys():
            assert k in target
            paras_pair.extend(_gen_paras_pair(src[k], target[k]))
    else:
        # for any other type, won't be handled
        pass
    return paras_pair


def _get_parameter_names(obj):
    parameter_names = []
    for attr in obj.__dict__:
        val = getattr(obj, attr)
        if isinstance(val, Network):
            parameter_names.extend(get_parameter_names(val))
        elif isinstance(val, LayerFunc):
            parameter_names.append(val.param_name)
            if val.bias_name is not None:
                parameter_names.append(val.bias_name)
        elif isinstance(val, tuple) or isinstance(val, list) or isinstance(
                val, set):
            for x in val:
                parameter_names.extend(get_parameter_names(x))
        elif isinstance(val, dict):
            for x in list(val.values()):
                parameter_names.extend(get_parameter_names(x))
        else:
            # for any other type, won't be handled
            pass
    return parameter_names


class Network(object):
    """
    A Network is an unordered set of LayerFuncs or Networks.
    """

    def sync_paras_to(self, target_net, gpu_id=0, decay=0.0):
        """
        Args:
            target_net: Network object deepcopy from source network
            gpu_id: gpu id of target_net 
            decay: Float. The decay to use. 
                   target_net_weights = decay * target_net_weights + (1 - decay) * source_net_weights
        """
        assert not target_net is self, "cannot copy between identical networks"
        assert isinstance(target_net, Network)
        assert self.__class__.__name__ == target_net.__class__.__name__, \
            "must be the same class for para syncing!"
        assert (decay >= 0 and decay <= 1)

        paras_pair = _gen_paras_pair(self, target_net)

        place = fluid.CPUPlace() if gpu_id < 0 \
                else fluid.CUDAPlace(gpu_id)
        fluid_executor = fluid.Executor(place)
        sync_paras_program = fluid.Program()

        with fluid.program_guard(sync_paras_program):
            for (src_var_name, target_var_name, is_bias) in paras_pair:
                src_var = _fetch_framework_var(src_var_name, is_bias)
                target_var = _fetch_framework_var(target_var_name, is_bias)
                fluid.layers.assign(decay * target_var + (1 - decay) * src_var,
                                    target_var)
        fluid_executor.run(sync_paras_program)

    def get_parameter_names(self):
        return _get_parameter_names(self)


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
