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
import subprocess
import threading

from parl.remote.master import Master
from parl.remote.worker import Worker
from parl.remote.client import disconnect
from parl.remote import exceptions
from parl.utils import get_free_tcp_port


@parl.remote_class(wait=False, n_gpu=1)
class Actor(object):
    def __init__(self, arg1=None, arg2=None):
        self.arg1 = arg1
        self.arg2 = arg2

    def get_arg1(self):
        return self.arg1

    def get_arg2(self):
        return self.arg2

    def set_arg1(self, value):
        self.arg1 = value

    def set_arg2(self, value):
        self.arg2 = value

    def add_one(self, value):
        value += 1
        return value

    def add(self, x, y):
        time.sleep(3)
        return x + y

    def will_raise_exception_func(self):
        x = 1 / 0


class TestCluster(unittest.TestCase):
    def tearDown(self):
        disconnect()
        time.sleep(10)  # wait for test case finishing

    def test_actor_exception(self):
        port = get_free_tcp_port()
        master = Master(port=port, device='gpu')
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(3)
        worker1 = Worker('localhost:{}'.format(port), 0, None, "0")
        for _ in range(3):
            if master.gpu_num == 1:
                break
            time.sleep(10)
        self.assertEqual(1, master.gpu_num)
        parl.connect('localhost:{}'.format(port))

        with self.assertRaises(exceptions.FutureFunctionError):
            actor = Actor(abcd='a bug')
            actor.get_arg1()  # calling any function will raise an exception

        actor2 = Actor()
        for _ in range(3):
            if master.gpu_num == 0:
                break
            time.sleep(10)

        future_result = actor2.add_one(1)
        self.assertEqual(future_result.get(), 2)
        self.assertEqual(0, master.gpu_num)

        master.exit()
        worker1.exit()

    def test_actor_exception_2(self):
        port = get_free_tcp_port()
        master = Master(port=port, device='gpu')
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(3)
        worker1 = Worker('localhost:{}'.format(port), 0, None, "0")
        self.assertEqual(1, master.gpu_num)
        parl.connect('localhost:{}'.format(port))
        actor = Actor()
        with self.assertRaises(exceptions.FutureFunctionError):
            future_object = actor.will_raise_exception_func()
            future_object.get()  # raise exception

        actor2 = Actor()
        for _ in range(5):
            if master.gpu_num == 0:
                break
            time.sleep(10)
        future_result = actor2.add_one(1)
        self.assertEqual(future_result.get(), 2)
        self.assertEqual(0, master.gpu_num)
        del actor
        del actor2
        worker1.exit()
        master.exit()


if __name__ == '__main__':
    unittest.main()
