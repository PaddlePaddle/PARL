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
from parl.remote.job_center import JobCenter
import time
import threading
from parl.remote.client import disconnect
from parl.utils import logger, get_free_tcp_port


class TestCluster(unittest.TestCase):
    def test_worker_run(self):
        port = get_free_tcp_port()
        master = Master(port=port, xpu='gpu')
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(3)
        worker = Worker('localhost:{}'.format(port), 0, None, 2)
        worker_th = threading.Thread(target=worker.run)
        worker_th.start()

        for _ in range(2):
            if master.gpu_num == 2:
                break
            time.sleep(5)
        self.assertEqual(2, master.gpu_num)

        assert worker_th.is_alive()

        master.exit()
        worker.exit()

    def test_cpu_worker_connect_gpu_master(self):
        port = get_free_tcp_port()
        master = Master(port=port, xpu='gpu')
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(3)
        worker = Worker('localhost:{}'.format(port), 1, None, 0)
        worker_th = threading.Thread(target=worker.run)
        worker_th.start()
        for _ in range(2):
            if not worker.worker_is_alive:
                break
            time.sleep(5)
        self.assertEqual(master.gpu_num, 0)

        master.exit()
        worker.exit()

    def test_gpu_worker_connect_cpu_master(self):
        port = get_free_tcp_port()
        master = Master(port=port, xpu='cpu')
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(3)
        worker = Worker('localhost:{}'.format(port), 0, None, 2)
        worker_th = threading.Thread(target=worker.run)
        worker_th.start()
        for _ in range(2):
            if not worker.worker_is_alive:
                break
            time.sleep(5)
        self.assertEqual(worker.worker_is_alive, False)
        self.assertEqual(master.cpu_num, 0)

        master.exit()
        worker.exit()

    def test_gpu_worker_exit(self):
        port = get_free_tcp_port()
        master = Master(port=port, xpu='gpu')
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(3)
        job_center = JobCenter('localhost:{}'.format(port), 'gpu')
        worker = Worker('localhost:{}'.format(port), 0, None, 2)
        worker_th = threading.Thread(target=worker.run)
        worker_th.start()
        for _ in range(2):
            if master.gpu_num == 2:
                break
            time.sleep(5)
        self.assertEqual(master.gpu_num, 2)
        worker.exit()
        for _ in range(2):
            if master.gpu_num == 0:
                break
            time.sleep(10)
        self.assertEqual(master.gpu_num, 0)
        master.exit()

    def test_cpu_worker_exit(self):
        port = get_free_tcp_port()
        master = Master(port=port, xpu='cpu')
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(3)
        job_center = JobCenter('localhost:{}'.format(port), 'cpu')
        worker = Worker('localhost:{}'.format(port), 1, None, 0)
        worker_th = threading.Thread(target=worker.run)
        worker_th.start()
        time.sleep(3)
        self.assertEqual(master.cpu_num, 1)
        worker.exit()
        for _ in range(2):
            if master.cpu_num == 0:
                break
            time.sleep(10)
        self.assertEqual(master.cpu_num, 0)
        master.exit()


if __name__ == '__main__':
    unittest.main()
