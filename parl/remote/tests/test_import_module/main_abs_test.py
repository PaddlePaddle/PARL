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


class TestImport(unittest.TestCase):
    def tearDown(self):
        disconnect()

    def test_import_local_module(self):
        from Module2 import B
        port = 8448
        master = Master(port=port)
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(1)
        worker = Worker('localhost:{}'.format(port), 1)
        time.sleep(10)
        parl.connect("localhost:8448")
        obj = B()
        res = obj.add_sum(10, 5)
        self.assertEqual(res, 15)
        worker.exit()
        master.exit()


if __name__ == '__main__':
    unittest.main()
