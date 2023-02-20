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
import threading
from parl.remote import exceptions
from parl.remote.future_mode import FutureObject
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

    def get_arg1_after_sleep(self, sleep_seconds):
        time.sleep(sleep_seconds)
        return self.arg1


class TestFutureObject(XparlTestCase):
    def test_resulf_of_get_function(self):
        self.add_master()
        self.add_worker(n_cpu=1)
        parl.connect('localhost:{}'.format(self.port))
        actor = Actor(arg1=10, arg2=20)
        future_obj = actor.get_arg1()
        assert isinstance(future_obj, FutureObject)
        result = future_obj.get()
        assert result == 10

        future_obj = actor.get_arg2()
        assert isinstance(future_obj, FutureObject)
        result = future_obj.get()
        assert result == 20
        actor.destroy()

    def test_calling_get_function_twice(self):
        self.add_master()
        self.add_worker(n_cpu=1)
        parl.connect('localhost:{}'.format(self.port))
        actor = Actor()
        future_obj = actor.get_arg1()
        result = future_obj.get()
        with self.assertRaises(exceptions.FutureGetRepeatedlyError):
            result = future_obj.get()
        actor.destroy()

    def test_calling_get_function_with_block_false(self):
        self.add_master()
        self.add_worker(n_cpu=1)
        parl.connect('localhost:{}'.format(self.port))

        actor = Actor(arg1="arg1")
        sleep_seconds = 3
        future_obj = actor.get_arg1_after_sleep(sleep_seconds=sleep_seconds)
        with self.assertRaises(exceptions.FutureObjectEmpty):
            result = future_obj.get(block=False)

        time.sleep(sleep_seconds + 1)

        result = future_obj.get(block=False)
        assert result == "arg1"

        actor.destroy()

    def test_calling_get_nowait_function(self):
        self.add_master()
        self.add_worker(n_cpu=1)
        parl.connect('localhost:{}'.format(self.port))
        actor = Actor(arg1="arg1")
        sleep_seconds = 3
        future_obj = actor.get_arg1_after_sleep(sleep_seconds=sleep_seconds)
        with self.assertRaises(exceptions.FutureObjectEmpty):
            result = future_obj.get_nowait()

        time.sleep(sleep_seconds + 1)

        result = future_obj.get_nowait()
        assert result == "arg1"
        actor.destroy()

    def test_calling_get_function_with_timeout(self):
        self.add_master()
        self.add_worker(n_cpu=1)
        parl.connect('localhost:{}'.format(self.port))

        actor = Actor(arg1="arg1")
        sleep_seconds = 3
        future_obj = actor.get_arg1_after_sleep(sleep_seconds=sleep_seconds)
        with self.assertRaises(exceptions.FutureObjectEmpty):
            result = future_obj.get(timeout=1)

        result = future_obj.get(timeout=sleep_seconds + 1)
        assert result == "arg1"
        actor.destroy()


if __name__ == '__main__':
    unittest.main(failfast=True)
