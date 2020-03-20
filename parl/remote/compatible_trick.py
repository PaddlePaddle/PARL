#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
This file is used to fix the problem that cloudpickle cannot load some packages normally in Mac OS.
We hack the problem by trying load these packages in the main module in advance.

Template:

try:
    import [PACKAGE1]
except ImportError:
    pass

try:
    import [PACKAGE2]
except ImportError:
    pass

"""
from parl.utils import _IS_MAC

if _IS_MAC:
    try:
        import rlschool
    except ImportError:
        pass
