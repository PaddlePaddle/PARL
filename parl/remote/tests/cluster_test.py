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
import timeout_decorator
import subprocess


@parl.remote_class
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

    def get_unable_serialize_object(self):
        return UnableSerializeObject()

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
        #time.sleep(20)
        #command = ("pkill -f remote/job.py")
        #subprocess.call([command], shell=True)

    def test_actor_exception(self):
        master = Master(port=1235)
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(1)
        worker1 = Worker('localhost:1235', 1)
        self.assertEqual(1, master.cpu_num)
        parl.connect('localhost:1235')
        with self.assertRaises(exceptions.RemoteError):
            actor = Actor(abcd='a bug')

        actor2 = Actor()
        self.assertEqual(actor2.add_one(1), 2)
        self.assertEqual(0, master.cpu_num)

        master.exit()
        worker1.exit()

    @timeout_decorator.timeout(seconds=300)
    def test_actor_exception(self):
        master = Master(port=1236)
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(1)
        worker1 = Worker('localhost:1236', 1)
        self.assertEqual(1, master.cpu_num)
        parl.connect('localhost:1236')
        actor = Actor()
        try:
            actor.will_raise_exception_func()
        except:
            pass
        actor2 = Actor()
        time.sleep(30)
        self.assertEqual(actor2.add_one(1), 2)
        self.assertEqual(0, master.cpu_num)
        del actor
        del actor2
        worker1.exit()
        master.exit()

    def test_reset_actor(self):
        # start the master
        master = Master(port=1237)
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(1)

        worker1 = Worker('localhost:1237', 4)
        parl.connect('localhost:1237')
        for i in range(10):
            actor = Actor()
            ret = actor.add_one(1)
            self.assertEqual(ret, 2)
        del actor
        time.sleep(20)
        self.assertEqual(master.cpu_num, 4)
        worker1.exit()
        master.exit()

    def test_add_worker(self):
        master = Master(port=1234)
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(1)
        worker1 = Worker('localhost:1234', 4)
        self.assertEqual(master.cpu_num, 4)
        worker2 = Worker('localhost:1234', 4)
        self.assertEqual(master.cpu_num, 8)

        worker2.exit()
        time.sleep(30)
        self.assertEqual(master.cpu_num, 4)

        master.exit()
        worker1.exit()


if __name__ == '__main__':
    unittest.main()
