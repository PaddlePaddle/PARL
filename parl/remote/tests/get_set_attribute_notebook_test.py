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
import time
import threading
import random

from parl.utils import logger
from unittest import mock
from parl.utils.test_utils import XparlTestCase


@parl.remote_class
class Actor(object):
    def __init__(self, arg1, arg2, arg3, arg4):
        self.arg1 = arg1
        self.arg2 = arg2
        self.arg3 = arg3
        self.GLOBAL_CLIENT = arg4

    def arg1(self, x, y):
        time.sleep(0.2)
        return x + y

    def arg5(self):
        return 100

    def set_new_attr(self):
        self.new_attr_1 = 200


class Test_get_and_set_attribute(XparlTestCase):
    def test_get_attribute(self):
        return_true = mock.Mock(return_value=True)
        with mock.patch(
                'parl.remote.remote_class_serialization.is_implemented_in_notebook',
                return_true):
            self.add_master()
            self.add_worker(n_cpu=1)
            arg1 = np.random.randint(100)
            arg2 = np.random.randn()
            arg3 = np.random.randn(3, 3)
            arg4 = 100
            parl.connect('localhost:{}'.format(self.port))

            actor = Actor(arg1, arg2, arg3, arg4)

            self.assertTrue(arg1 == actor.arg1)
            self.assertTrue(arg2 == actor.arg2)
            self.assertTrue((arg3 == actor.arg3).all())
            self.assertTrue(arg4 == actor.GLOBAL_CLIENT)

    def test_set_attribute(self):
        return_true = mock.Mock(return_value=True)
        with mock.patch(
                'parl.remote.remote_class_serialization.is_implemented_in_notebook',
                return_true):
            self.add_master()
            self.add_worker(n_cpu=1)
            arg1 = 3
            arg2 = 3.5
            arg3 = np.random.randn(3, 3)
            arg4 = 100
            parl.connect('localhost:{}'.format(self.port))

            actor = Actor(arg1, arg2, arg3, arg4)
            actor.arg1 = arg1
            actor.arg2 = arg2
            actor.arg3 = arg3
            actor.GLOBAL_CLIENT = arg4
            self.assertTrue(arg1 == actor.arg1)
            self.assertTrue(arg2 == actor.arg2)
            self.assertTrue((arg3 == actor.arg3).all())
            self.assertTrue(arg4 == actor.GLOBAL_CLIENT)


if __name__ == '__main__':
    unittest.main(failfast=True)
