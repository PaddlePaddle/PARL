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

__version__ = "1.4"
"""
generates new PARL python API
"""
import os

from tensorboardX import SummaryWriter
from parl.utils.utils import _HAS_FLUID, _HAS_TORCH, _HAS_PADDLE
from parl.utils import logger

if 'PARL_BACKEND' in os.environ and os.environ['PARL_BACKEND'] != '':
    assert os.environ['PARL_BACKEND'] in ['fluid', 'paddle', 'torch']
    backend_dict = {
        'fluid': 'Static Graph-based PARL',
        'paddle': 'Dynamic Graph-based PARL',
        'torch': 'Torch-based PARL'
    }
    logger.info(
        'Have found environment variable `PARL_BACKEND`==\'{}\', switching framework to [{}]'
        .format(os.environ['PARL_BACKEND'],
                backend_dict[os.environ['PARL_BACKEND']]))
    if os.environ['PARL_BACKEND'] == 'fluid':
        assert _HAS_FLUID, 'Static Graph-based PARL requires paddlepaddle version <= 1.8.5'
        from parl.core.fluid import *
        from parl.core.fluid.plutils.compiler import compile
    elif os.environ['PARL_BACKEND'] == 'paddle':
        assert _HAS_PADDLE, 'Dynamic Graph-based PARL requires paddlepaddle version >= 2.0.0'
        from parl.core.paddle import *
    elif os.environ['PARL_BACKEND'] == 'torch':
        assert _HAS_TORCH, 'Torch-based PARL requires torch, which is not installed.'
        from parl.core.torch import *
else:
    if _HAS_FLUID:
        from parl.core.fluid import *
        from parl.core.fluid.plutils.compiler import compile
    elif _HAS_PADDLE:
        from parl.core.paddle import *
    elif _HAS_TORCH:
        from parl.core.torch import *

from parl.remote import remote_class, connect
from parl import algorithms
