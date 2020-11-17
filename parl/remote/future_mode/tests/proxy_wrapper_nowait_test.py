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
from parl.utils import logger, get_free_tcp_port
from parl.remote.master import Master
from parl.remote.worker import Worker
from parl.remote.exceptions import FutureFunctionError
import time
import threading
import random


@parl.remote_class(wait=False)
class Actor(object):
    def __init__(self, arg):
        self.arg = arg


@parl.remote_class(wait=False)
class ActorReservedAttr1(object):
    def __init__(self):
        self.xparl_remote_wrapper_obj = 0

    def get_arg(self):
        return 0


@parl.remote_class(wait=False)
class ActorReservedAttr2(object):
    def __init__(self):
        self.xparl_remote_wrapper_calling_queue = 0

    def get_arg(self):
        return 0


@parl.remote_class(wait=False)
class ActorReservedAttr3(object):
    def __init__(self):
        self.xparl_remote_wrapper_internal_lock = 0

    def get_arg(self):
        return 0


@parl.remote_class(wait=False)
class ActorReservedAttr4(object):
    def __init__(self):
        self.xparl_calling_finished_event = 0

    def get_arg(self):
        return 0


@parl.remote_class(wait=False)
class ActorReservedAttr5(object):
    def __init__(self):
        self.xparl_remote_object_exception = 0

    def get_arg(self):
        return 0


class Test_proxy_wrapper(unittest.TestCase):
    def tearDown(self):
        disconnect()

    def test_kwargs_with_reserved_names_1(self):
        port = get_free_tcp_port()
        master = Master(port=port)
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(3)
        worker1 = Worker('localhost:{}'.format(port), 1)

        parl.connect('localhost:{}'.format(port))

        with self.assertRaises(AssertionError):
            actor = Actor(__xparl_proxy_wrapper_nowait__=1)

        master.exit()
        worker1.exit()

    def test_kwargs_with_reserved_names_2(self):
        port = get_free_tcp_port()
        master = Master(port=port)
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(3)
        worker1 = Worker('localhost:{}'.format(port), 1)

        parl.connect('localhost:{}'.format(port))

        with self.assertRaises(AssertionError):
            actor = Actor(__xparl_remote_class__=1)

        master.exit()
        worker1.exit()

    def test_kwargs_with_reserved_names_3(self):
        port = get_free_tcp_port()
        master = Master(port=port)
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(3)
        worker1 = Worker('localhost:{}'.format(port), 1)

        parl.connect('localhost:{}'.format(port))

        with self.assertRaises(AssertionError):
            actor = Actor(__xparl_remote_class_max_memory__=1)

        master.exit()
        worker1.exit()

    def test_attribute_with_reserved_names_1(self):
        port = get_free_tcp_port()
        master = Master(port=port)
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(3)
        worker1 = Worker('localhost:{}'.format(port), 1)

        parl.connect('localhost:{}'.format(port))

        actor = ActorReservedAttr1()  # will raise error in another thread
        with self.assertRaises(FutureFunctionError):
            actor.get_arg()  # calling any function will raise error

        master.exit()
        worker1.exit()

    def test_attribute_with_reserved_names_2(self):
        port = get_free_tcp_port()
        master = Master(port=port)
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(3)
        worker1 = Worker('localhost:{}'.format(port), 1)

        parl.connect('localhost:{}'.format(port))

        actor = ActorReservedAttr2()  # will raise error in another thread
        with self.assertRaises(FutureFunctionError):
            actor.get_arg()  # calling any function will raise error

        master.exit()
        worker1.exit()

    def test_attribute_with_reserved_names_3(self):
        port = get_free_tcp_port()
        master = Master(port=port)
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(3)
        worker1 = Worker('localhost:{}'.format(port), 1)

        parl.connect('localhost:{}'.format(port))

        actor = ActorReservedAttr3()  # will raise error in another thread
        with self.assertRaises(FutureFunctionError):
            actor.get_arg()  # calling any function will raise error

        master.exit()
        worker1.exit()

    def test_attribute_with_reserved_names_4(self):
        port = get_free_tcp_port()
        master = Master(port=port)
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(3)
        worker1 = Worker('localhost:{}'.format(port), 1)

        parl.connect('localhost:{}'.format(port))

        actor = ActorReservedAttr4()  # will raise error in another thread
        with self.assertRaises(FutureFunctionError):
            actor.get_arg()  # calling any function will raise error

        master.exit()
        worker1.exit()

    def test_attribute_with_reserved_names_5(self):
        port = get_free_tcp_port()
        master = Master(port=port)
        th = threading.Thread(target=master.run)
        th.start()
        time.sleep(3)
        worker1 = Worker('localhost:{}'.format(port), 1)

        parl.connect('localhost:{}'.format(port))

        actor = ActorReservedAttr5()  # will raise error in another thread
        with self.assertRaises(FutureFunctionError):
            actor.get_arg()  # calling any function will raise error

        master.exit()
        worker1.exit()


if __name__ == '__main__':
    unittest.main()
