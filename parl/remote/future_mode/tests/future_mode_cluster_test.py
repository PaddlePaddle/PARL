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
from parl.remote import exceptions
from parl.utils.test_utils import XparlTestCase

@parl.remote_class(wait=False)
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


class TestCluster(XparlTestCase):
    def test_actor_exception_2(self):
        self.add_master()
        self.add_worker(n_cpu=1)
        parl.connect('localhost:{}'.format(self.port))
        actor = Actor()
        with self.assertRaises(exceptions.FutureFunctionError):
            future_object = actor.will_raise_exception_func()
            future_object.get()  # raise exception

        actor2 = Actor()
        future_result = actor2.add_one(1)
        self.assertEqual(future_result.get(), 2)
        actor.destroy()
        actor2.destroy()

if __name__ == '__main__':
    unittest.main(failfast=True)
