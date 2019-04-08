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
from parl.utils.scheduler import PiecewiseScheduler


class TestScheduler(unittest.TestCase):
    def test_PiecewiseScheduler_with_multi_values(self):
        scheduler = PiecewiseScheduler([(0, 0.1), (3, 0.2), (7, 0.3)])
        for i in range(10):
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

    def test_PiecewiseScheduler_with_empty(self):
        try:
            scheduler = PiecewiseScheduler([])
        except AssertionError:
            # expected
            return
        assert False

    def test_PiecewiseScheduler_with_incorrect_steps(self):
        try:
            scheduler = PiecewiseScheduler([(10, 0.1), (1, 0.2)])
        except AssertionError:
            # expected
            return
        assert False


if __name__ == '__main__':
    unittest.main()
