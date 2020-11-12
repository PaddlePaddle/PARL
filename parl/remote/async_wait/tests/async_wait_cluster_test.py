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
from parl.remote import exceptions
import subprocess
from parl.utils import logger


@parl.remote_class(wait=False)
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

    def test_actor_exception(self):
        port = 8535
        logger.info("running:test_actor_exception")
        master = Master(port=port)
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(3)
        worker1 = Worker('localhost:{}'.format(port), 1)
        for _ in range(3):
            if master.cpu_num == 1:
                break
            time.sleep(10)
        self.assertEqual(1, master.cpu_num)
        logger.info("running:test_actor_exception: 0")
        parl.connect('localhost:{}'.format(port))
        logger.info("running:test_actor_exception: 1")

        with self.assertRaises(exceptions.AsyncFunctionError):
            actor = Actor(abcd='a bug')
            actor.get_arg1()  # calling any function will raise an exception
        logger.info("running:test_actor_exception: 2")

        actor2 = Actor()
        for _ in range(3):
            if master.cpu_num == 0:
                break
            time.sleep(10)

        future_result = actor2.add_one(1)
        self.assertEqual(future_result.get(), 2)
        self.assertEqual(0, master.cpu_num)

        master.exit()
        worker1.exit()

    def test_actor_exception_2(self):
        port = 8536
        logger.info("running: test_actor_exception_2")
        master = Master(port=port)
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(3)
        worker1 = Worker('localhost:{}'.format(port), 1)
        self.assertEqual(1, master.cpu_num)
        parl.connect('localhost:{}'.format(port))
        actor = Actor()
        with self.assertRaises(exceptions.AsyncFunctionError):
            future_object = actor.will_raise_exception_func()
            future_object.get()  # raise exception

        actor2 = Actor()
        for _ in range(5):
            if master.cpu_num == 0:
                break
            time.sleep(10)
        future_result = actor2.add_one(1)
        self.assertEqual(future_result.get(), 2)
        self.assertEqual(0, master.cpu_num)
        del actor
        del actor2
        worker1.exit()
        master.exit()

    def test_reset_actor(self):
        port = 8537
        logger.info("running: test_reset_actor")
        # start the master
        master = Master(port=port)
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(3)

        worker1 = Worker('localhost:{}'.format(port), 4)
        parl.connect('localhost:{}'.format(port))
        for _ in range(10):
            actor = Actor()
            future_result = actor.add_one(1)
            self.assertEqual(future_result.get(), 2)
        del actor

        for _ in range(10):
            if master.cpu_num == 4:
                break
            time.sleep(10)

        self.assertEqual(master.cpu_num, 4)
        worker1.exit()
        master.exit()


if __name__ == '__main__':
    unittest.main()
