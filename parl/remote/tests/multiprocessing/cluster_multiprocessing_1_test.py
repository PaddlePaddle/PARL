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
import time
import threading
import multiprocessing

from parl.remote.master import Master
from parl.remote.worker import Worker
from parl.remote.client import disconnect
from parl.utils import get_free_tcp_port, _IS_WINDOWS


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
    def _connect_and_create_actor(cluster_addr):
        parl.connect(cluster_addr)
        for _ in range(2):
            actor = Actor()
            ret = actor.add_one(1)
            assert ret == 2
        disconnect()

    def _create_actor(self):
        for _ in range(2):
            actor = Actor()
            ret = actor.add_one(1)
            self.assertEqual(ret, 2)

    def _create_master(self, port, exit_event):
        master = Master(port=port)
        th = threading.Thread(target=master.run)
        th.start()

        exit_event.wait()
        master.exit()
        th.join()

    def _create_worker(self, port, exit_event):
        worker = Worker('localhost:{}'.format(port), 4)
        th = threading.Thread(target=worker.run)
        th.start()

        exit_event.wait()
        worker.exit()
        th.join()

    def test_connect_and_create_actor_in_multiprocessing_with_connected_in_main_process(
            self):
        if _IS_WINDOWS:
            # In windows, creating master in multiprocessing will raise the error:
            #`TypeError: cannot serialize '_io.TextIOWrapper' object`
            return

        port = get_free_tcp_port()

        # start the master
        master_exit_event = multiprocessing.Event()
        master_proc = multiprocessing.Process(
            target=self._create_master, args=(port, master_exit_event))
        master_proc.start()

        time.sleep(1)

        # start the worker
        worker_exit_event = multiprocessing.Event()
        worker_proc = multiprocessing.Process(
            target=self._create_worker, args=(port, worker_exit_event))
        worker_proc.start()

        proc1 = multiprocessing.Process(
            target=self._connect_and_create_actor,
            args=('localhost:{}'.format(port), ))
        proc2 = multiprocessing.Process(
            target=self._connect_and_create_actor,
            args=('localhost:{}'.format(port), ))
        proc1.start()
        proc2.start()

        proc1.join()
        proc2.join()

        # make sure that the client of the main process still works
        parl.connect('localhost:{}'.format(port))
        self._create_actor()

        worker_exit_event.set()
        master_exit_event.set()
        worker_proc.join()
        master_proc.join()


if __name__ == '__main__':
    unittest.main()
