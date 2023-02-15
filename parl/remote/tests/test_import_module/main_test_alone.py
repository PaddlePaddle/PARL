#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import unittest
import parl
import time
import threading
from parl.remote.master import Master
from parl.remote.worker import Worker
from parl.remote.client import disconnect
from parl.utils import get_free_tcp_port
import time


class TestImport(unittest.TestCase):
    def tearDown(self):
        disconnect()
        time.sleep(1)

    def test_import_local_module(self):
        from Module2 import B
        port = get_free_tcp_port()
        master = Master(port=port)
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(1)
        worker = Worker('localhost:{}'.format(port), 1)
        time.sleep(10)
        parl.connect("localhost:{}".format(port))
        obj = B()
        res = obj.add_sum(10, 5)
        self.assertEqual(res, 15)
        worker.exit()
        master.exit()

    def test_import_subdir_module_0(self):
        from subdir import Module
        port = get_free_tcp_port()
        master = Master(port=port)
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(1)
        worker = Worker('localhost:{}'.format(port), 1)
        time.sleep(10)
        parl.connect(
            "localhost:{}".format(port),
            distributed_files=[
                os.path.join('subdir', 'Module.py'),
                os.path.join('subdir', '__init__.py')
            ])
        obj = Module.A()
        res = obj.add_sum(10, 5)
        self.assertEqual(res, 15)
        worker.exit()
        master.exit()

    def test_import_subdir_module_1(self):
        from subdir.Module import A
        port = get_free_tcp_port()
        master = Master(port=port)
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(1)
        worker = Worker('localhost:{}'.format(port), 1)
        time.sleep(10)
        parl.connect(
            "localhost:{}".format(port),
            distributed_files=[
                os.path.join('subdir', 'Module.py'),
                os.path.join('subdir', '__init__.py')
            ])
        obj = A()
        res = obj.add_sum(10, 5)
        self.assertEqual(res, 15)
        worker.exit()
        master.exit()


if __name__ == '__main__':
    unittest.main()
