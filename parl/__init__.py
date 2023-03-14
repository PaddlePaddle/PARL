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

__version__ = "2.2.1"
"""
generates new PARL python API
"""
import os
from parl.utils.utils import _HAS_FLUID, _HAS_TORCH, _HAS_PADDLE, _IS_WINDOWS
from parl.utils import logger

if 'XPARL_igonre_core' not in os.environ: # load the core module by default
    if 'PARL_BACKEND' in os.environ and os.environ['PARL_BACKEND'] != '':
        assert os.environ['PARL_BACKEND'] in ['fluid', 'paddle', 'torch']
        logger.info(
            'Have found environment variable `PARL_BACKEND`==\'{}\', switching backend framework to [{}]'
            .format(os.environ['PARL_BACKEND'], os.environ['PARL_BACKEND']))
        if os.environ['PARL_BACKEND'] == 'paddle':
            from parl.core.paddle import *
        elif os.environ['PARL_BACKEND'] == 'fluid':
            from parl.core.fluid import *
            from parl.core.fluid.plutils.compiler import compile
        elif os.environ['PARL_BACKEND'] == 'torch':
            assert _HAS_TORCH, 'Torch-based PARL requires torch, which is not installed.'
            from parl.core.torch import *
    else:
        if _HAS_PADDLE:
            # disable the signal handler inside paddle, which shows signal information when we kill the job in xparl.
            import paddle
            paddle.disable_signal_handler()
            from parl.core.paddle import *
            if _HAS_TORCH:
                logger.info("PARL detects two backend frameworks: paddle, torch. Use paddle by default.")
                logger.info("To use torch as backend, `export PARL_BACKEND=torch` before running the scripts.")
        elif _HAS_FLUID:
            from parl.core.fluid import *
            from parl.core.fluid.plutils.compiler import compile
        elif _HAS_TORCH:
            from parl.core.torch import *
    from parl import algorithms

if not _IS_WINDOWS:
    from parl.remote import remote_class, connect
