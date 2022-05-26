#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
import unittest
from parl.remote.utils import get_xparl_python


class TestUtils(unittest.TestCase):
    def test_get_default_xparl_python(self):
        xparl_python = get_xparl_python()
        default_python = sys.executable.split()
        assert len(default_python) == len(xparl_python)
        for i, x in enumerate(xparl_python):
            assert x == default_python[i]

    def test_get_specified_xparl_python(self):
        specified_python = "/opt/compiler/gcc-10/lib/ld-linux-x86-64.so.2 --library-path /opt/compiler/gcc-10/lib:~/miniconda3/envs/py36/lib: ~/miniconda3/envs/py36/bin/python"
        os.environ['XPARL_PYTHON'] = specified_python

        xparl_python = get_xparl_python()

        specified_python = specified_python.split()
        assert len(specified_python) == len(xparl_python)
        for i, x in enumerate(xparl_python):
            assert x == specified_python[i]


if __name__ == '__main__':
    unittest.main()
