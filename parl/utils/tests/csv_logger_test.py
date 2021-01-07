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
import unittest
from parl.utils import CSVLogger


class TestCSVLogger(unittest.TestCase):
    def test_log_dict(self):
        tmp_file = "test_log_dict_tmp.csv"
        csv_logger = CSVLogger(tmp_file)

        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]

        for x in data:
            csv_logger.log_dict(x)

        csv_logger.close()

        with open(tmp_file, "r") as f:
            lines = [x.strip() for x in f.readlines()]
            assert lines == ['a,b', '1,2', '3,4']

        os.remove(tmp_file)

    def test_log_dict_with_different_keys(self):
        tmp_file = "test_log_dict_with_different_keys_tmp.csv"
        csv_logger = CSVLogger(tmp_file)

        data1 = {"a": 1, "b": 2}
        data2 = {"a": 3, "c": 4}

        csv_logger.log_dict(data1)

        with self.assertRaises(AssertionError):
            csv_logger.log_dict(data2)

        csv_logger.close()

        os.remove(tmp_file)


if __name__ == '__main__':
    unittest.main()
