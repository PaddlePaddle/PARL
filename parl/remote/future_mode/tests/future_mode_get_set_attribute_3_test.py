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
from parl.utils.test_utils import XparlTestCase

@parl.remote_class(wait=False)
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
    def test_non_existing_attribute_same_with_existing_method(self):
        self.add_master()
        self.add_worker(n_cpu=1)
        arg1 = np.random.randint(100)
        arg2 = np.random.randn()
        arg3 = np.random.randn(3, 3)
        arg4 = 100
        parl.connect('localhost:{}'.format(self.port))
        actor = Actor(arg1, arg2, arg3, arg4)
        actor.new_attr_2 = 300
        self.assertEqual(300, actor.new_attr_2)
        actor.set_new_attr()
        self.assertEqual(200, actor.new_attr_1)
        self.assertTrue(callable(actor.arg5))
        def call_non_existing_method():
            return actor.arg2(10)

        self.assertRaises(TypeError, call_non_existing_method)
        actor.destroy()


if __name__ == '__main__':
    unittest.main(failfast=True)
