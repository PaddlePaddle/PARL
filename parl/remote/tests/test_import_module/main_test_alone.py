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
import parl
import multiprocessing as mp
from parl.utils.test_utils import XparlTestCase
import os

class TestImport(XparlTestCase):
    def test_import_local_module(self):
        from Module2 import B
        self.add_master()
        self.add_worker(n_cpu=1)
        parl.connect("localhost:{}".format(self.port))
        obj = B()
        res = obj.add_sum(10, 5)
        self.assertEqual(res, 15)

    def test_import_subdir_module_0(self):
        from subdir import Module
        self.add_master()
        self.add_worker(n_cpu=1)
        parl.connect(
            "localhost:{}".format(self.port),
            distributed_files=[os.path.join('subdir', 'Module.py'),
                               os.path.join('subdir', '__init__.py')])
        obj = Module.A()
        res = obj.add_sum(10, 5)
        self.assertEqual(res, 15)

    def test_import_subdir_module_1(self):
        from subdir.Module import A
        self.add_master()
        self.add_worker(n_cpu=1)
        parl.connect(
            "localhost:{}".format(self.port),
            distributed_files=[os.path.join('subdir', 'Module.py'),
                               os.path.join('subdir', '__init__.py')])
        obj = A()
        res = obj.add_sum(10, 5)
        self.assertEqual(res, 15)

if __name__ == '__main__':
    unittest.main(failfast=True)
