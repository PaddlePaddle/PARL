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


@parl.remote_class(n_gpus=2)
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
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
        import torch
        assert (torch.cuda.device_count() == 4)


class TestCluster(unittest.TestCase):
    def tearDown(self):
        disconnect()

    def test_actor_exception_1(self):
        port = get_free_tcp_port()
        logger.info("running:test_actor_exception")
        master = Master(port=port, xpu='gpu')
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(3)
        worker1 = Worker('localhost:{}'.format(port), 0, None, 4)
        for _ in range(3):
            if master.gpu_num == 4:
                break
            time.sleep(10)
        self.assertEqual(4, master.gpu_num)
        parl.connect('localhost:{}'.format(port))

        with self.assertRaises(exceptions.RemoteError):
            actor1 = Actor(abcd='a bug')
        for _ in range(3):
            if master.gpu_num == 4:
                break
            time.sleep(10)
        self.assertEqual(4, master.gpu_num)

        actor2 = Actor()
        for _ in range(3):
            if master.gpu_num == 2:
                break
            time.sleep(10)
        self.assertEqual(actor2.add_one(1), 2)
        self.assertEqual(2, master.gpu_num)
        del actor2
        for _ in range(3):
            if master.gpu_num == 4:
                break
            time.sleep(10)
        self.assertEqual(4, master.gpu_num)

        master.exit()
        worker1.exit()

    def test_actor_exception_2(self):
        port = get_free_tcp_port()
        master = Master(port=port, xpu='gpu')
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(3)
        worker1 = Worker('localhost:{}'.format(port), 0, None, 4)
        for _ in range(3):
            if master.gpu_num == 4:
                break
            time.sleep(10)
        self.assertEqual(4, master.gpu_num)

        parl.connect('localhost:{}'.format(port))
        actor1 = Actor()
        with self.assertRaises(exceptions.RemoteError):
            actor1.will_raise_exception_func()
        actor2 = Actor()
        for _ in range(5):
            if master.gpu_num == 2:
                break
            time.sleep(10)
        self.assertEqual(actor2.add_one(1), 2)
        self.assertEqual(2, master.gpu_num)
        del actor1
        del actor2
        for _ in range(5):
            if master.gpu_num == 4:
                break
            time.sleep(10)
        self.assertEqual(4, master.gpu_num)
        worker1.exit()
        master.exit()

    def test_cuda_visible_devices_setting(self):
        port = get_free_tcp_port()
        master = Master(port=port, xpu='gpu')
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(3)
        os.environ['PARL_BACKEND'] = 'torch'
        worker1 = Worker('localhost:{}'.format(port), 0, None, 4)
        for _ in range(3):
            if master.gpu_num == 4:
                break
            time.sleep(10)
        self.assertEqual(4, master.gpu_num)

        parl.connect('localhost:{}'.format(port))
        actor = Actor()
        with self.assertRaises(exceptions.RemoteError):
            actor.assert_device_count_fail()
        for _ in range(5):
            if master.gpu_num == 4:
                break
            time.sleep(10)
        self.assertEqual(4, master.gpu_num)
        worker1.exit()
        master.exit()



if __name__ == '__main__':
    unittest.main()
