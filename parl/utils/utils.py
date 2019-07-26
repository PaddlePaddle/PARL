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

import sys

__all__ = [
    'has_func', 'action_mapping', 'to_str', 'to_byte', 'is_PY2', 'is_PY3',
    'MAX_INT32', '_HAS_FLUID'
]


def has_func(obj, fun):
    """check if a class has specified function: https://stackoverflow.com/a/5268474

    Args:
        obj: the class to check
        fun: specified function to check
    Returns:
        A bool to indicate if obj has funtion "fun"
    """
    check_fun = getattr(obj, fun, None)
    return callable(check_fun)


def action_mapping(model_output_act, low_bound, high_bound):
    """ mapping action space [-1, 1] of model output 
        to new action space [low_bound, high_bound].

    Args:
        model_output_act: np.array, which value is in [-1, 1]
        low_bound: float, low bound of env action space
        high_bound: float, high bound of env action space

    Returns:
        action: np.array, which value is in [low_bound, high_bound]
    """
    assert high_bound > low_bound
    action = low_bound + (model_output_act - (-1.0)) * (
        (high_bound - low_bound) / 2.0)
    return action


def to_str(byte):
    """ convert byte to string in pytohn2/3
    """
    return str(byte.decode())


def to_byte(string):
    """ convert byte to string in pytohn2/3
    """
    return string.encode()


def is_PY2():
    return sys.version_info[0] == 2


def is_PY3():
    return sys.version_info[0] == 3


def get_fluid_version():
    import paddle
    fluid_version = int(paddle.__version__.replace('.', ''))
    return fluid_version


MAX_INT32 = 0x7fffffff

try:
    from paddle import fluid
    fluid_version = get_fluid_version()
    assert fluid_version >= 151, "PARL requires paddle>=1.5.1"
    _HAS_FLUID = True
except ImportError:
    _HAS_FLUID = False
