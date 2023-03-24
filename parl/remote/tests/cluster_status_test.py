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
from parl.remote.monitor import ClusterMonitor
from parl.utils.test_utils import XparlTestCase


@parl.remote_class
class Actor(object):
    def __init__(self, x=10):
        self.x = x
        self.data = []

class TestClusterStatus(XparlTestCase):
    def test_cluster_status(self):
        master = Master(port=self.port)
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(5)
        parl.connect('localhost:{}'.format(self.port))
        self.add_worker(n_cpu=1)
        time.sleep(40)
        status_info = master.cluster_monitor.get_status_info()
        self.assertEqual(status_info, 'has 0 used cpus, 1 vacant cpus, 0 used_gpus, 0 vacant_gpus.')
        actor = Actor()
        time.sleep(40)
        status_info = master.cluster_monitor.get_status_info()
        self.assertEqual(status_info, 'has 1 used cpus, 0 vacant cpus, 0 used_gpus, 0 vacant_gpus.')
        master.exit()
        th.join()


if __name__ == '__main__':
    unittest.main()
