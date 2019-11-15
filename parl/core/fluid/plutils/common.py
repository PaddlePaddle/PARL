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
from parl.utils import machine_info

__all__ = ['fetch_framework_var', 'fetch_value', 'set_value', 'inverse']


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
    if core_var == None:
        raise KeyError(
            "Unable to find the variable:{}. Synchronize paramsters before initialization or attr_name does not exist."
            .format(attr_name))
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


def set_value(attr_name, value, is_gpu_available):
    """ Given name of ParamAttr, set numpy value to the parameter in global_scope
    
    Args:
        attr_name(string): ParamAttr name of parameter
        value(np.array): numpy value
        is_gpu_available(bool): whether is gpu available
    """
    place = fluid.CUDAPlace(0) if is_gpu_available else fluid.CPUPlace()
    var = _fetch_var(attr_name, return_numpy=False)
    var.set(value, place)


def inverse(x):
    """ Inverse 0/1 variable

    Args:
        x: variable with float32 dtype
    
    Returns:
        inverse_x: variable with float32 dtype
    """
    inverse_x = -1.0 * x + 1.0
    return inverse_x
