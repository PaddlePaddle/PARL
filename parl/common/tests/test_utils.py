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
import unittest
from parl.common.error_handling import LastExpError
from parl.common.utils import concat_dicts, split_dict


class TestUtils(unittest.TestCase):
    def test_concat_and_split_dict(self):
        d1 = dict(
            sensors=np.random.rand(3, 10),
            states=np.random.rand(3, 20),
            actions=[[1, 2, 3], [4, 5], [6]])
        d2 = dict(
            sensors=np.random.rand(2, 10),
            states=np.random.rand(2, 20),
            actions=[[7, 8, 9, 10], [11, 12]])
        D, starts = concat_dicts([d1, d2])
        self.assertEqual(starts, [0, 3, 5])
        for k in ["sensors", "actions"]:
            self.assertTrue(np.array_equal(D[k][0:3], d1[k]))
            self.assertTrue(np.array_equal(D[k][3:], d2[k]))
        self.assertTrue(np.array_equal(D["states"][0:3], d1["states"]))
        self.assertTrue(np.array_equal(D["states"][3:], d2["states"]))

        dd1, dd2 = split_dict(D, starts)
        self.assertEqual(dd1.viewkeys(), dd2.viewkeys())
        for k in dd1.iterkeys():
            if k == "actions":
                self.assertEqual(dd1[k], d1[k])
                self.assertEqual(dd2[k], d2[k])
            else:
                self.assertTrue(np.array_equal(dd1[k], d1[k]))
                self.assertTrue(np.array_equal(dd2[k], d2[k]))
        with self.assertRaises(Exception):
            d3 = dict(
                sensors=np.random.rand(3, 10),
                states=np.random.rand(2, 20),
                actions=[[7, 8, 9, 10], [11, 12]])
            concat_dicts([d1, d3])
        with self.assertRaises(Exception):
            d3 = dict(
                sensors=np.random.rand(3, 10),
                states=np.random.rand(3, 20),
                actions=[[7, 8, 9, 10]])
            concat_dicts([d1, d3])


if __name__ == '__main__':
    unittest.main()
