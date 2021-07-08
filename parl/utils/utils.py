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
from parl.utils import logger

__all__ = [
    'has_func', 'to_str', 'to_byte', '_IS_PY2', '_IS_PY3', 'MAX_INT32',
    '_HAS_FLUID', '_HAS_PADDLE', '_HAS_TORCH', '_IS_WINDOWS', '_IS_MAC',
    'kill_process', 'get_fluid_version', 'isnotebook', 'check_version_for_xpu',
    'check_version_for_fluid', 'check_model_method'
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


def to_str(byte):
    """ convert byte to string in pytohn2/3
    """
    return str(byte.decode())


def to_byte(string):
    """ convert byte to string in pytohn2/3
    """
    return string.encode()


_IS_PY2 = (sys.version_info[0] == 2)
_IS_PY3 = (sys.version_info[0] == 3)


def get_fluid_version():
    import paddle
    paddle_version = int(paddle.__version__.replace('.', '').split('-')[0])
    return paddle_version


MAX_INT32 = 0x7fffffff

try:
    _HAS_FLUID = False
    _HAS_PADDLE = False
    import paddle
    from paddle import fluid

    paddle_version = get_fluid_version()
    logger.info("paddlepaddle version: {}.".format(paddle.__version__))
    if paddle_version < 200 and paddle_version != 0:
        assert paddle_version >= 185, "PARL requires paddle >= 1.8.5 and paddle < 2.0.0"
        _HAS_FLUID = True
    else:
        _HAS_PADDLE = True
except ImportError as e:
    _HAS_FLUID = False
    _HAS_PADDLE = False
    if _IS_PY2 and "{}".format(e) != "No module named paddle":
        logger.warning("import paddle error:\n{}".format(e))
    if _IS_PY3 and "{}".format(e) != "No module named 'paddle'":
        logger.warning("import paddle error:\n{}".format(e))

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
        with os.popen(command) as p:
            p.read()
    else:
        command = "ps aux | grep {} | awk '{{print $2}}' | xargs kill -9".format(
            regex_pattern)
        subprocess.call([command], shell=True)


def isnotebook():
    """check if the code is excuted in the IPython notebook
    Reference: https://stackoverflow.com/a/39662359
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def check_version_for_xpu():
    """check paddle version if the code requires to run on xpu
    """
    err = "To use xpu, PARL requires paddle version >= 2.0.0. " \
          "Please make sure the version is consistent with your code."
    import paddle
    from paddle import fluid

    paddle_version = get_fluid_version()
    assert paddle_version >= 200 or paddle_version == 0, err


def check_version_for_fluid():
    """check the paddle version if the code requires fluid-based implementation.
    """
    err = "To use fluid version of examples, PARL requires paddle version < 2.0.0. " \
          "Please make sure the version is consistent with your code."
    import paddle
    from paddle import fluid

    paddle_version = get_fluid_version()
    assert paddle_version >= 185 and paddle_version < 200, err


def check_model_method(model, method, algo):
    """ check method existence for input model to algo

    Args:
        model(parl.Model): model for checking
        method(str): method name
        algo(str): algorithm name

    Raises:
        AssertionError: if method is not implemented in model
    """
    if method == 'forward':
        # check if forward is overridden by the subclass
        assert callable(
            getattr(model, 'forward',
                    None)), "forward should be a function in model class"
        assert model.forward.__func__ is not super(
            model.__class__, model
        ).forward.__func__, "{}'s model needs to implement forward method. \n".format(
            algo)
    else:
        # check if the specified method is implemented
        assert hasattr(model, method) and callable(
            getattr(
                model, method,
                None)), "{}'s model needs to implement {} method. \n".format(
                    algo, method)
