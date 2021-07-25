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
from parl.utils import get_free_tcp_port
from unittest import mock


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
        time.sleep(60)  # wait for test case finishing

    def test_actor_exception(self):
        return_true = mock.Mock(return_value=True)
        with mock.patch(
                'parl.remote.remote_class_serialization.is_implemented_in_notebook',
                return_true):
            port = get_free_tcp_port()
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

            with self.assertRaises(exceptions.RemoteError):
                actor = Actor(abcd='a bug')
            logger.info("running:test_actor_exception: 2")

            actor2 = Actor()
            for _ in range(3):
                if master.cpu_num == 0:
                    break
                time.sleep(10)
            self.assertEqual(actor2.add_one(1), 2)
            self.assertEqual(0, master.cpu_num)

            master.exit()
            worker1.exit()


if __name__ == '__main__':
    unittest.main()
