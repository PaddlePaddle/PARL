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

import random
import unittest
import parl
import numpy as np
from parl.remote.client import disconnect
from parl.utils import logger, get_free_tcp_port
from parl.remote.master import Master
from parl.remote.worker import Worker
import time
import threading
import random


@parl.remote_class(n_gpus=1)
class Actor1(object):
    def __init__(self, arg):
        self.arg = arg

    def add_one(self, x):
        return x + 1


@parl.remote_class(n_gpus=1)
class Actor2(object):
    def __init__(self):
        self._xparl_remote_wrapper_obj = 0


class Test_proxy_wrapper(unittest.TestCase):
    def tearDown(self):
        disconnect()

    def test_proxy_wrapper_wait(self):
        port = get_free_tcp_port()
        master = Master(port, None, 'gpu')
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(1)
        worker1 = Worker('localhost:{}'.format(port), 0, None, "0")
        worker_th = threading.Thread(target=worker1.run)
        worker_th.start()

        parl.connect('localhost:{}'.format(port))

        actor = Actor1(10)

        assert actor.arg == 10

        assert actor.add_one(1) == 2

        actor = None

        master.exit()
        worker1.exit()

    def test_kwargs_with_reserved_names(self):
        port = get_free_tcp_port()
        master = Master(port, None, 'gpu')
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(1)
        worker1 = Worker('localhost:{}'.format(port), 0, None, "0")
        worker_th = threading.Thread(target=worker1.run)
        worker_th.start()

        parl.connect('localhost:{}'.format(port))

        with self.assertRaises(AssertionError):
            actor = Actor1(_xparl_proxy_wrapper_nowait__=1)
            actor = None

        master.exit()
        worker1.exit()

    def test_attribute_with_reserved_names(self):
        port = get_free_tcp_port()
        master = Master(port, None, 'gpu')
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(1)
        worker1 = Worker('localhost:{}'.format(port), 0, None, "0")
        worker_th = threading.Thread(target=worker1.run)
        worker_th.start()

        parl.connect('localhost:{}'.format(port))

        with self.assertRaises(AssertionError):
            actor = Actor2()
            actor = None

        master.exit()
        worker1.exit()

    def test_get_original_class(self):
        origin_class = Actor1._original
        origin_actor = origin_class(10)

        assert origin_actor.arg == 10
        assert origin_actor.add_one(1) == 2


if __name__ == '__main__':
    unittest.main()
