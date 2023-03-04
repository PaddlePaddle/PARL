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

import unittest
import parl
from parl.remote.master import Master
from parl.remote.worker import Worker
import time
import threading
from parl.remote.client import disconnect
from parl.utils import logger, get_free_tcp_port
from parl.utils.test_utils import XparlTestCase


class TestCluster(XparlTestCase):
    def test_worker_run(self):
        master = Master(port=self.port, device='gpu')
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(10)
        self.add_worker(n_cpu=0, gpu="0,1")

        for _ in range(10):
            if master.gpu_num == 2:
                break
            time.sleep(5)
        self.assertEqual(2, master.gpu_num)

        master.exit()

    def test_cpu_worker_connect_gpu_master(self):
        master = Master(port=self.port, device='gpu')
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(10)
        self.add_worker(n_cpu=1, gpu="")
        self.assertEqual(master.gpu_num, 0)
        self.assertEqual(master.cpu_num, 0)

        master.exit()

    def test_gpu_worker_connect_cpu_master(self):
        master = Master(port=self.port, device='cpu')
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(10)
        self.add_worker(n_cpu=0, gpu="0,1")

        self.assertEqual(master.cpu_num, 0)
        self.assertEqual(master.gpu_num, 0)

        master.exit()

    def test_gpu_worker_exit(self):
        master = Master(port=self.port, device='gpu')
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(10)
        self.add_worker(n_cpu=0, gpu="0,1")

        for _ in range(10):
            if master.gpu_num == 2:
                break
            time.sleep(5)
        self.assertEqual(master.gpu_num, 2)
        self.remove_all_workers()

        for _ in range(10):
            if master.gpu_num == 0:
                break
            time.sleep(10)
        self.assertEqual(master.gpu_num, 0)

        master.exit()

    def test_cpu_worker_exit(self):
        master = Master(port=self.port, device='cpu')
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(10)
        self.add_worker(n_cpu=1)
        time.sleep(10)
        self.assertEqual(master.cpu_num, 1)
        self.remove_all_workers()
        for _ in range(10):
            if master.cpu_num == 0:
                break
            time.sleep(10)
        self.assertEqual(master.cpu_num, 0)
        master.exit()


if __name__ == '__main__':
    unittest.main(failfast=True)
