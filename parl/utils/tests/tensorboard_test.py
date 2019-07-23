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
from parl.utils import tensorboard
import numpy as np


class TestUtils(unittest.TestCase):
    def tearDown(self):
        tensorboard.flush()

    def test_add_scalar(self):
        x = range(100)
        for i in x:
            tensorboard.add_scalar('y=2x', i * 2, i)

    def test_add_histogram(self):
        for i in range(10):
            x = np.random.random(1000)
            tensorboard.add_histogram('distribution centers', x + i, i)


if __name__ == '__main__':
    unittest.main()
