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
from parl.remote.exceptions import FutureFunctionError
from parl.remote.future_mode.future_object import FutureObject
from parl.utils.test_utils import XparlTestCase
import time
import threading
import random


@parl.remote_class(wait=False)
class Actor(object):
    def __init__(self, arg):
        self.arg = arg

    def add_one(self, x):
        return x + 1


@parl.remote_class(wait=False)
class Actor2(object):
    def __init__(self):
        self._xparl_remote_wrapper_obj = 0

    def get_arg(self):
        return 0


class Test_proxy_wrapper(XparlTestCase):
    def test_proxy_wrapper_nowait(self):
        self.add_master()
        self.add_worker(n_cpu=1)
        parl.connect('localhost:{}'.format(self.port))
        actor = Actor(10)

        assert actor.arg == 10

        actor.arg2 = 20
        assert actor.arg2 == 20

        future_obj = actor.add_one(1)
        assert isinstance(future_obj, FutureObject)
        assert future_obj.get() == 2
        actor.destroy()

    def test_kwargs_with_reserved_names_1(self):
        self.add_master()
        self.add_worker(n_cpu=1)
        parl.connect('localhost:{}'.format(self.port))

        with self.assertRaises(AssertionError):
            actor = Actor(_xparl_proxy_wrapper_nowait__=1)

    def test_attribute_with_reserved_names_1(self):
        self.add_master()
        self.add_worker(n_cpu=1)
        parl.connect('localhost:{}'.format(self.port))

        actor = Actor2()  # will raise error in another thread
        with self.assertRaises(FutureFunctionError):
            actor.get_arg()  # calling any function will raise error
        actor.destroy()

    def test_get_original_class(self):
        origin_class = Actor._original
        origin_actor = origin_class(10)

        assert origin_actor.arg == 10
        assert origin_actor.add_one(1) == 2


if __name__ == '__main__':
    unittest.main(failfast=True)
