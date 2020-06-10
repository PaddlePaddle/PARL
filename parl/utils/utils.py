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
import os
import subprocess
import numpy as np

__all__ = [
    'has_func', 'action_mapping', 'to_str', 'to_byte', 'is_PY2', 'is_PY3',
    'MAX_INT32', '_HAS_FLUID', '_HAS_TORCH', '_IS_WINDOWS', '_IS_MAC',
    'kill_process'
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
    assert np.all(((model_output_act<=1.0), (model_output_act>=-1.0))), \
        'the action should be in range [-1.0, 1.0]'
    assert high_bound > low_bound
    action = low_bound + (model_output_act - (-1.0)) * (
        (high_bound - low_bound) / 2.0)
    action = np.clip(action, low_bound, high_bound)
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
    assert fluid_version >= 161 or fluid_version == 0, "PARL requires paddle>=1.6.1"
    _HAS_FLUID = True
except ImportError:
    _HAS_FLUID = False

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

_IS_WINDOWS = (sys.platform == 'win32')
_IS_MAC = (sys.platform == 'darwin')


def kill_process(regex_pattern):
    """kill process whose execution commnad is matched by regex pattern

    Args:
        regex_pattern(string): regex pattern used to filter the process to be killed
    
    NOTE:
        In windows, we will replace sep `/` with `\\\\`
    """
    if _IS_WINDOWS:
        regex_pattern = regex_pattern.replace('/', '\\\\')
        command = r'''for /F "skip=2 tokens=2 delims=," %a in ('wmic process where "commandline like '%{}%'" get processid^,status /format:csv') do taskkill /F /T /pid %a'''.format(
            regex_pattern)
        os.popen(command).read()
    else:
        command = "ps aux | grep {} | awk '{{print $2}}' | xargs kill -9".format(
            regex_pattern)
        subprocess.call([command], shell=True)
