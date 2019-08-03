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

__version__ = "1.1.1"
"""
generates new PARL python API
"""

# trick to solve importing error
from tensorboardX import SummaryWriter

from parl.utils.utils import _HAS_FLUID

if _HAS_FLUID:
    from parl.core.fluid import *
    from parl.core.fluid.plutils.compiler import compile
else:
    print(
        "WARNING:PARL: Failed to import paddle. Only APIs for parallelization are available."
    )

from parl.remote import remote_class, RemoteManager
from parl import algorithms
