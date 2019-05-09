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

import numpy as np
import time
import unittest
from parl.utils.communication import dumps_return, loads_return


class TestCommunication(unittest.TestCase):
    def test_speed_of_dumps_loads_return(self):
        data1 = {
            i: {
                100 * str(j): [set(range(100)) for _ in range(10)]
                for j in range(100)
            }
            for i in range(100)
        }

        data2 = [np.random.RandomState(0).randn(100, 300)] * 5

        data3 = [np.random.RandomState(0).randn(400, 100, 3000)]

        for i, data in enumerate([data1, data2, data3]):
            start = time.time()
            for _ in range(10):
                serialize_bytes = dumps_return(data)
                deserialize_result = loads_return(serialize_bytes)
            print('Case {}, Average dump and load return time:'.format(i),
                  (time.time() - start) / 10)

    def test_speed_of_dumps_loads_argument(self):
        data1 = {
            i: {
                100 * str(j): [set(range(100)) for _ in range(10)]
                for j in range(100)
            }
            for i in range(100)
        }

        data2 = [np.random.RandomState(0).randn(100, 300)] * 5

        data3 = [np.random.RandomState(0).randn(400, 100, 3000)]

        for i, data in enumerate([data1, data2, data3]):
            start = time.time()
            for _ in range(10):
                serialize_bytes = dumps_return(data)
                deserialize_result = loads_return(serialize_bytes)
            print('Case {}, Average dump and load argument time:'.format(i),
                  (time.time() - start) / 10)


if __name__ == '__main__':
    unittest.main()
