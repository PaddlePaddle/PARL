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

import os
import parl
import unittest
import time
from parl.remote.client import get_global_client
from parl.utils.test_utils import XparlTestCase

@parl.remote_class(max_memory=350)
class Actor(object):
    def __init__(self, x=10):
        self.x = x
        self.data = []

    def add_500mb(self):
        self.data.append(os.urandom(500 * 1024**2))
        self.x += 1
        return self.x

class TestMaxMemory(XparlTestCase):
    def test_max_memory(self):
        self.add_master()
        self.add_worker(n_cpu=1)
        cluster_addr = 'localhost:{}'.format(self.port)
        time.sleep(5)
        parl.connect(cluster_addr)
        actor = Actor()
        time.sleep(30)
        self.assertEqual(1, get_global_client().actor_num.value)
        del actor
        actor1 = Actor()
        actor1.add_500mb()
        time.sleep(60)
        self.assertEqual(0, get_global_client().actor_num.value)


if __name__ == '__main__':
    unittest.main(failfast=True)
