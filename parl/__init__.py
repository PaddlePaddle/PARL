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
generates new PARL python API
"""

from parl.utils.utils import _HAS_FLUID

if _HAS_FLUID:
    from parl.framework import *
else:
    print(
        "WARNING:PARL: Failed to import paddle. Only APIs for parallelization are available."
    )

from parl.remote import remote_class, RemoteManager
