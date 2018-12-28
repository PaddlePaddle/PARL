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
from paddle.fluid.executor import _fetch_var
from parl.layers.layer_wrappers import LayerFunc
from parl.framework.model_base import Network

__all__ = [
    'fetch_framework_var', 'fetch_value', 'get_parameter_pairs',
    'get_parameter_names'
]


def fetch_framework_var(attr_name):
    """ Fetch framework variable according given attr_name.
        Return a new reusing variable through create_parameter way 

    Args:
        attr_name: string, attr name of parameter

    Returns:
        framework_var: framework.Varialbe
    """

    scope = fluid.executor.global_scope()
    core_var = scope.find_var(attr_name)
    shape = core_var.get_tensor().shape()
    framework_var = fluid.layers.create_parameter(
        shape=shape, dtype='float32', attr=fluid.ParamAttr(name=attr_name))
    return framework_var


def fetch_value(attr_name):
    """ Given name of ParamAttr, fetch numpy value of the parameter in global_scope
    
    Args:
        attr_name: ParamAttr name of parameter

    Returns:
        numpy.ndarray
    """
    return _fetch_var(attr_name, return_numpy=True)


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
        src_attrs = src.attr_holder.sorted()
        target_attrs = target.attr_holder.sorted()
        assert len(src_attrs) == len(target_attrs), \
                "number of ParamAttr between source layer and target layer should be same."
        for (src_attr, target_attr) in zip(src_attrs, target_attrs):
            if src_attr:
                assert target_attr, "ParamAttr between source layer and target layer is inconsistent."
                param_pairs.append((src_attr.name, target_attr.name))
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
            for attr in val.attr_holder.tolist():
                if attr:
                    parameter_names.append(attr.name)
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
