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
import numpy as np
from parl.remote.client import disconnect
from parl.utils import logger
from parl.remote.master import Master
from parl.remote.worker import Worker
import time
import threading
import random

@parl.remote_class
class Actor(object):
    def __init__(self, arg1=0, arg2=1.5, arg3=np.zeros((3, 3))):
        self.arg1 = arg1
        self.arg2 = arg2
        self.arg3 = arg3

    def get_arg1(self):
        return self.arg1

    def get_arg2(self):
        return self.arg2

    def get_arg3(self):
        return self.arg3

    def add(self, x, y):
        time.sleep(0.2)
        return x + y


class Test_get_and_set_attribute(unittest.TestCase):
    def tearDown(self):
        disconnect()

    def test_get_attribute(self):
        port1 = random.randint(6100, 6200)
        logger.info("running:test_get_attirbute")
        master = Master(port=port1)
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(3)
        worker1 = Worker('localhost:{}'.format(port1), 1)
        arg1 = np.random.randint(100)
        arg2 = np.random.randn()
        arg3 = np.random.randn(3, 3)
        parl.connect('localhost:{}'.format(port1))
        actor = Actor(arg1, arg2, arg3)
        self.assertTrue(arg1 == actor.arg1)
        self.assertTrue(arg2 == actor.arg2)
        self.assertTrue((arg3 == actor.arg3).all())
        master.exit()
        worker1.exit()

    def test_set_attribute(self):
        port2 = random.randint(6200, 6300)
        logger.info("running:test_set_attirbute")
        master = Master(port=port2)
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(3)
        worker1 = Worker('localhost:{}'.format(port2), 1)
        arg1 = 3
        arg2 = 3.5
        arg3 = np.random.randn(3, 3)
        parl.connect('localhost:{}'.format(port2))
        actor = Actor()
        actor.arg1 = arg1
        actor.arg2 = arg2
        actor.arg3 = arg3
        self.assertTrue(arg1 == actor.arg1)
        self.assertTrue(arg2 == actor.arg2)
        self.assertTrue((arg3 == actor.arg3).all())
        master.exit()
        worker1.exit()

if __name__ == '__main__':
    unittest.main()
