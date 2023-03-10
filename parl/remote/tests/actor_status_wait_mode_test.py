#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from parl.utils.test_utils import XparlTestCase
import time
from parl.remote.exceptions import RemoteError, FutureFunctionError

@parl.remote_class(wait=False)
class Actor(object):
    def __init__(self, arg1=None, arg2=None):
        self.arg1 = arg1
        self.arg2 = arg2

    def add(self, x, y):
        return x + y
    
    def exit(self):
        import sys
        sys.exit(0)

class TestActorStatus(XparlTestCase):
    def test_default_mode(self):
        self.add_master()
        self.add_worker(n_cpu=1)
        parl.connect("localhost:{}".format(self.port))
        actor = Actor()
        res = actor.add(10, 5).get()
        self.assertEqual(res, 15)
        with self.assertRaises(FutureFunctionError):
            res = actor.exit().get()
        with self.assertRaises(FutureFunctionError):
            future = actor.add(10, 5)

if __name__ == '__main__':
    unittest.main(failfast=True)
