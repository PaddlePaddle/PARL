#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
os.environ['XPARL'] = 'True'
import parl
import unittest


@parl.remote_class(max_memory=350)
class Actor(object):
    def __init__(self, x=10):
        self.x = x
        self.data = []

    def add_500mb(self):
        self.data.append(os.urandom(500 * 1024**2))
        self.x += 1
        return self.x


class TestLocalActor(unittest.TestCase):
    def test_create_actors_without_pre_connection(self):
        actor = Actor()


if __name__ == '__main__':
    unittest.main()
