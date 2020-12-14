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
from parl.core.model_base import ModelBase


class TestBaseModel(ModelBase):
    def forward(self, x, y):
        return x + y


class ModelBaseTest(unittest.TestCase):
    def setUp(self):
        self.model = TestBaseModel()

    def test_forward(self):
        x, y = 10, 20
        expected_out = x + y
        forward_out = self.model(x, y)
        self.assertEqual(forward_out, expected_out)


if __name__ == '__main__':
    unittest.main()
