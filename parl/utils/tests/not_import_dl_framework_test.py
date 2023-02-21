#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import sys
import os
import importlib

class TestNotImportPaddle(unittest.TestCase):
    def test_import(self):
        # setting this enviroment variable will not import deep learning framework
        os.environ['XPARL_igonre_core'] = 'true'
        import parl
        self.assertFalse('paddle' in sys.modules)
        # remove the environment vaiable and reimport the lib
        del os.environ['XPARL_igonre_core']
        importlib.reload(parl)
        self.assertTrue('paddle' in sys.modules)

if __name__ == '__main__':
    unittest.main()
