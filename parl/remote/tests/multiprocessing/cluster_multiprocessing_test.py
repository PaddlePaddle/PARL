#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0 #
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import parl
import time
import threading
import multiprocessing

from parl.remote.master import Master
from parl.remote.worker import Worker
from parl.remote.client import disconnect
from parl.utils import _IS_WINDOWS


@parl.remote_class
class Actor(object):
    def __init__(self, arg1=None, arg2=None):
        self.arg1 = arg1
        self.arg2 = arg2

    def add_one(self, value):
        value += 1
        return value


class TestCluster(unittest.TestCase):
    def tearDown(self):
        disconnect()

    #In windows, multiprocessing.Process cannot run the method of class, but static method is ok.
    @staticmethod
    def _create_actor():
        for _ in range(2):
            actor = Actor()
            ret = actor.add_one(1)
            assert ret == 2

    def test_create_actor_in_multiprocessing(self):
        # start the master
        master = Master(port=8240)
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(1)

        worker1 = Worker('localhost:8240', 4)
        parl.connect('localhost:8240')

        if not _IS_WINDOWS:  # In windows, fork process cannot access client created in main process.
            proc1 = multiprocessing.Process(target=self._create_actor)
            proc2 = multiprocessing.Process(target=self._create_actor)
            proc1.start()
            proc2.start()

            proc1.join()
            proc2.join()
            print("[test_create_actor_in_multiprocessing]  Join")

        # make sure that the client of the main process still works
        self._create_actor()

        worker1.exit()
        master.exit()


if __name__ == '__main__':
    unittest.main()
