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
import numpy as np
from parl.utils.scheduler import *


class TestScheduler(unittest.TestCase):
    def test_PiecewiseScheduler_with_multi_values(self):
        scheduler = PiecewiseScheduler([(0, 0.1), (3, 0.2), (7, 0.3)])
        for i in range(1, 11):
            value = scheduler.step()
            if i < 3:
                assert value == 0.1
            elif i < 7:
                assert value == 0.2
            else:
                assert value == 0.3

    def test_PiecewiseScheduler_with_one_value(self):
        scheduler = PiecewiseScheduler([(0, 0.1)])
        for i in range(10):
            value = scheduler.step()
            assert value == 0.1

        scheduler = PiecewiseScheduler([(3, 0.1)])
        for i in range(10):
            value = scheduler.step()
            assert value == 0.1

    def test_PiecewiseScheduler_with_step_num(self):
        scheduler = PiecewiseScheduler([(0, 0.1), (3, 0.2), (7, 0.3)])

        value = scheduler.step()
        assert value == 0.1

        value = scheduler.step(2)
        assert value == 0.2

        value = scheduler.step(4)
        assert value == 0.3

    def test_PiecewiseScheduler_with_empty(self):
        with self.assertRaises(AssertionError):
            scheduler = PiecewiseScheduler([])

    def test_PiecewiseScheduler_with_incorrect_steps(self):
        with self.assertRaises(AssertionError):
            tscheduler = PiecewiseScheduler([(10, 0.1), (1, 0.2)])

    def test_LinearDecayScheduler(self):
        scheduler = LinearDecayScheduler(start_value=10, max_steps=10)
        for i in range(10):
            value = scheduler.step()
            np.testing.assert_almost_equal(value, 10 - (i + 1), 8)

        for i in range(5):
            value = scheduler.step()
            np.testing.assert_almost_equal(value, 0, 8)

    def test_LinearDecayScheduler_with_step_num(self):
        scheduler = LinearDecayScheduler(start_value=10, max_steps=10)
        value = scheduler.step(5)
        np.testing.assert_almost_equal(value, 5, 8)

        value = scheduler.step(3)
        np.testing.assert_almost_equal(value, 2, 8)


if __name__ == '__main__':
    unittest.main()
