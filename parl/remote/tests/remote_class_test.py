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

from parl.remote.master import Master
from parl.remote.worker import Worker
from parl.remote.client import disconnect
from parl.utils import get_free_tcp_port


@parl.remote_class
class Actor(object):
    def __init__(self):
        pass


class TestCluster(unittest.TestCase):
    def tearDown(self):
        disconnect()

    def test_class_decorated_by_remote_class(self):
        port = get_free_tcp_port()
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
        parl.connect('localhost:{}'.format(port))

        actor = Actor()

        for _ in range(3):
            if master.cpu_num == 0:
                break
            time.sleep(10)
        self.assertEqual(0, master.cpu_num)

        master.exit()
        worker1.exit()

    def test_function_decorated_by_remote_class(self):
        port = get_free_tcp_port()
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
        parl.connect('localhost:{}'.format(port))

        with self.assertRaises(AssertionError):

            @parl.remote_class
            def func():
                pass

        self.assertEqual(1, master.cpu_num)

        master.exit()
        worker1.exit()

    def test_passing_arguments_with_unsupported_argument_names(self):
        with self.assertRaises(AssertionError):

            @parl.remote_class(xxx=10)
            class Actor2(object):
                def __init__(self):
                    pass


if __name__ == '__main__':
    unittest.main()
