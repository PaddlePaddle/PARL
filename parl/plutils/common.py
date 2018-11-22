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
Common functions of PARL framework
"""

import paddle.fluid as fluid
from parl.layers.layer_wrappers import LayerFunc
from parl.framework.model_base import Network

__all__ = ['fetch_framework_var', 'get_parameter_pairs', 'get_parameter_names']


def fetch_framework_var(attr_name, is_bias):
    """ Fetch framework variable according given attr_name.
        Return a new reusing variable through create_parameter way 

    Args:
        attr_name: string, attr name of parameter
        is_bias: bool, decide whether the parameter is bias
    Returns:
        framework_var: framework.Varialbe
    """

    scope = fluid.executor.global_scope()
    core_var = scope.find_var(attr_name)
    shape = core_var.get_tensor().shape()
    framework_var = fluid.layers.create_parameter(
        shape=shape,
        dtype='float32',
        attr=fluid.ParamAttr(name=attr_name),
        is_bias=is_bias)
    return framework_var


def get_parameter_pairs(src, target):
    """ Recursively get pairs of parameter names between src and target

    Args:
        src: parl.Network/parl.LayerFunc/list/tuple/set/dict
        target: parl.Network/parl.LayerFunc/list/tuple/set/dict
    Returns:
        param_pairs: list of all tuple(src_param_name, target_param_name, is_bias)
                     between src and target
    """

    param_pairs = []
    if isinstance(src, Network):
        for attr in src.__dict__:
            if not attr in target.__dict__:
                continue
            src_var = getattr(src, attr)
            target_var = getattr(target, attr)
            param_pairs.extend(get_parameter_pairs(src_var, target_var))
    elif isinstance(src, LayerFunc):
        param_pairs.append((src.param_attr.name, target.param_attr.name,
                            False))
        if src.bias_attr:
            param_pairs.append((src.bias_attr.name, target.bias_attr.name,
                                True))
    elif isinstance(src, tuple) or isinstance(src, list) or isinstance(
            src, set):
        for src_var, target_var in zip(src, target):
            param_pairs.extend(get_parameter_pairs(src_var, target_var))
    elif isinstance(src, dict):
        for k in src.keys():
            assert k in target
            param_pairs.extend(get_parameter_pairs(src[k], target[k]))
    else:
        # for any other type, won't be handled
        pass
    return param_pairs


def get_parameter_names(obj):
    """ Recursively get parameter names in obj,
        mainly used to get parameter names of a parl.Network

    Args:
        obj: parl.Network/parl.LayerFunc/list/tuple/set/dict
    Returns:
        parameter_names: list of string, all parameter names in obj
    """

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
