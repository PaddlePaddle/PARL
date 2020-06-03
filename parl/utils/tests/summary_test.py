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
from parl.utils import summary
import numpy as np
from parl.utils import logger
import os


class TestUtils(unittest.TestCase):
    def tearDown(self):
        if hasattr(summary, 'flush'):
            summary.flush()

    def test_add_scalar(self):
        x = range(100)
        for i in x:
            summary.add_scalar('y=2x', i * 2, i)
        self.assertTrue(os.path.exists('./train_log/summary_test'))

    def test_add_histogram(self):
        if not hasattr(summary, 'add_histogram'):
            return
        for i in range(10):
            x = np.random.random(1000)
            summary.add_histogram('distribution centers', x + i, i)


if __name__ == '__main__':
    logger.auto_set_dir(action='d')
    unittest.main()
