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
from parl.utils.test_utils import XparlTestCase


@parl.remote_class(n_gpu=1)
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

    def assert_device_count_fail(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        import paddle
        assert (paddle.device.cuda.device_count() == 2)


class TestCluster(XparlTestCase):
    def test_actor_exception_1(self):
        self.add_master(device="gpu")
        self.add_worker(n_cpu=0, gpu="0,1")
        port = self.port
        parl.connect('localhost:{}'.format(port))

        with self.assertRaises(exceptions.RemoteError):
            actor = Actor(abcd='a bug')

        actor1 = Actor()
        self.assertEqual(actor1.add_one(1), 2)

        actor2 = Actor()
        self.assertEqual(actor2.add(1, 2), 3)

    def test_actor_exception_2(self):
        self.add_master(device="gpu")
        self.add_worker(n_cpu=0, gpu="0,1")
        port = self.port
        parl.connect('localhost:{}'.format(port))

        actor1 = Actor()
        with self.assertRaises(exceptions.RemoteError):
            actor1.will_raise_exception_func()

        actor2 = Actor()
        self.assertEqual(actor2.add_one(1), 2)

        del actor1
        del actor2

        actor3 = Actor()
        self.assertEqual(actor3.add_one(1), 2)

        actor4 = Actor()
        self.assertEqual(actor3.add_one(1), 2)

    def test_cuda_visible_devices_setting(self):
        self.add_master(device="gpu")
        self.add_worker(n_cpu=0, gpu="0,1")
        port = self.port
        parl.connect('localhost:{}'.format(port))

        actor1 = Actor()
        self.assertEqual(actor1.add_one(1), 2)

        actor2 = Actor()
        with self.assertRaises(exceptions.RemoteError):
            actor2.assert_device_count_fail()

        actor3 = Actor()
        self.assertEqual(actor3.add_one(1), 2)


if __name__ == '__main__':
    unittest.main()
