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
import unittest
from parl.utils import logger
import parl
from parl.remote.client import disconnect
from parl.remote.master import Master
from parl.remote.worker import Worker
import time
import threading

c = 10
port = 3002
if __name__ == '__main__':
    master = Master(port=port)
    th = threading.Thread(target=master.run)
    th.setDaemon(True)
    th.start()
time.sleep(5)
cluster_addr = 'localhost:{}'.format(port)
parl.connect(cluster_addr)
worker = Worker(cluster_addr, 1)


@parl.remote_class
class Actor(object):
    def add(self, a, b):
        return a + b + c


actor = Actor()


class TestRecursive_actor(unittest.TestCase):
    def tearDown(self):
        disconnect()

    def test_global_running(self):
        self.assertEqual(actor.add(1, 2), 13)
        master.exit()
        worker.exit()


if __name__ == '__main__':
    unittest.main()
