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
from parl.utils import logger
from parl.utils.test_utils import XparlTestCase

@parl.remote_class
class Actor(object):
    def __init__(self, arg1=None, arg2=None):
        self.arg1 = arg1
        self.arg2 = arg2

    def add(self, x, y):
        return x + y

class TestCluster(XparlTestCase):
    def test_reset_actor(self):
        self.add_master()
        self.add_worker(n_cpu=4)
        parl.connect('localhost:{}'.format(self.port))
        for _ in range(10):
            actor = Actor()
            ret = actor.add(1, 1)
            self.assertEqual(ret, 2)
        del actor

        actors = []
        for i in range(4):
            actors.append(Actor())
        for i in range(4):
            ret = actors[i].add(2, 3)
            self.assertEqual(5, ret)

if __name__ == '__main__':
    unittest.main(failfast=True)
