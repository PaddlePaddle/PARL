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
import threading
import unittest
from parl.remote.communication import dumps_return, loads_return, \
        dumps_argument, loads_argument


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
            for _ in range(5):
                serialize_bytes = dumps_return(data)
                deserialize_result = loads_return(serialize_bytes)
            print('Case {}, Average dump and load return time:'.format(i),
                  (time.time() - start) / 5)

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
            for _ in range(5):
                serialize_bytes = dumps_argument(data)
                deserialize_result = loads_argument(serialize_bytes)
            print('Case {}, Average dump and load argument time:'.format(i),
                  (time.time() - start) / 5)

    def test_dumps_loads_return_with_custom_class(self):
        class A(object):
            def __init__(self):
                self.a = 3

        a = A()
        serialize_bytes = dumps_return(a)
        deserialize_result = loads_return(serialize_bytes)

        assert deserialize_result.a == 3

    def test_dumps_loads_argument_with_custom_class(self):
        class A(object):
            def __init__(self):
                self.a = 3

        a = A()
        serialize_bytes = dumps_argument(a)
        deserialize_result = loads_argument(serialize_bytes)

        assert deserialize_result[0][0].a == 3

    def test_dumps_loads_return_with_multi_thread(self):
        class A(object):
            def __init__(self, a):
                self.a = a

        def run(i):
            a = A(i)
            serialize_bytes = dumps_return(a)
            deserialize_result = loads_return(serialize_bytes)
            assert deserialize_result.a == i

        threads = []
        for i in range(50):
            t = threading.Thread(target=run, args=(i, ))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

    def test_dumps_loads_argument_with_multi_thread(self):
        class A(object):
            def __init__(self, a):
                self.a = a

        def run(i):
            a = A(i)
            serialize_bytes = dumps_argument(a)
            deserialize_result = loads_argument(serialize_bytes)
            assert deserialize_result[0][0].a == i

        threads = []
        for i in range(50):
            t = threading.Thread(target=run, args=(i, ))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()


if __name__ == '__main__':
    unittest.main()
