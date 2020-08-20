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

import os
import parl
import unittest
import time
import threading

from parl.remote.master import Master
from parl.remote.worker import Worker
from parl.remote.client import disconnect
from parl.remote.monitor import ClusterMonitor

from multiprocessing import Process


@parl.remote_class(max_memory=350)
class Actor(object):
    def __init__(self, x=10):
        self.x = x
        self.data = []

    def add_500mb(self):
        self.data.append(os.urandom(500 * 1024**2))
        self.x += 1
        return self.x


from parl.utils import logger


class TestMaxMemory(unittest.TestCase):
    def tearDown(self):
        disconnect()

    #In windows, multiprocessing.Process cannot run the method of class, but static method is ok.
    @staticmethod
    def actor(cluster_addr):
        parl.connect(cluster_addr)
        actor1 = Actor()
        time.sleep(10)
        actor1.add_500mb()

    def test_max_memory(self):
        port = 3001
        master = Master(port=port)
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(5)
        cluster_addr = 'localhost:{}'.format(port)
        worker = Worker(cluster_addr, 1)
        cluster_monitor = ClusterMonitor(cluster_addr)
        time.sleep(5)
        parl.connect(cluster_addr)
        actor = Actor()
        time.sleep(20)
        self.assertEqual(1, cluster_monitor.data['clients'][0]['actor_num'])
        del actor
        time.sleep(10)
        p = Process(target=self.actor, args=(cluster_addr, ))
        p.start()

        for _ in range(6):
            x = cluster_monitor.data['clients'][0]['actor_num']
            if x == 0:
                break
            else:
                time.sleep(10)
        if x == 1:
            raise ValueError("Actor max memory test failed.")
        self.assertEqual(0, cluster_monitor.data['clients'][0]['actor_num'])
        p.terminate()

        worker.exit()
        master.exit()


if __name__ == '__main__':
    unittest.main()
