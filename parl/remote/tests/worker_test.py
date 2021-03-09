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


class TestCluster(unittest.TestCase):
    def test_worker_run(self):
        port = get_free_tcp_port()
        master = Master(port=port)
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(3)
        worker = Worker('localhost:{}'.format(port), 1)
        worker_th = threading.Thread(target=worker.run)
        worker_th.start()

        for _ in range(3):
            if master.cpu_num == 1:
                break
            time.sleep(10)
        self.assertEqual(1, master.cpu_num)

        time.sleep(20)
        assert worker_th.is_alive()

        master.exit()
        worker.exit()


if __name__ == '__main__':
    unittest.main()
