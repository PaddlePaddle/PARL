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
from parl.remote.async_wait import FutureObject


@parl.remote_class(wait=False)
class Actor(object):
    def __init__(self, arg1=None, arg2=None):
        self.arg1 = arg1
        self.arg2 = arg2

    def get_arg1(self):
        return self.arg1

    def get_arg2(self):
        return self.arg2


class TestFutureObject(unittest.TestCase):
    def tearDown(self):
        disconnect()

    def test_resulf_of_get_function(self):
        port = 8635
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

        actor = Actor(arg1=10, arg2=20)

        future_obj = actor.get_arg1()
        assert isinstance(future_obj, FutureObject)
        result = future_obj.get()
        assert result == 10

        future_obj = actor.get_arg2()
        assert isinstance(future_obj, FutureObject)
        result = future_obj.get()
        assert result == 20

        master.exit()
        worker1.exit()

    def test_calling_get_function_twice(self):
        port = 8636
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
        future_obj = actor.get_arg1()
        result = future_obj.get()
        with self.assertRaises(exceptions.FutureGetRepeatedlyError):
            result = future_obj.get()

        master.exit()
        worker1.exit()


if __name__ == '__main__':
    unittest.main()
