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


class ModelBaseTest(unittest.TestCase):
    def setUp(self):
        self.model = ModelBase()

    def test_set_and_get_model_id(self):
        model_id = 'id1'
        self.model.set_model_id(model_id)
        self.assertEqual(model_id, self.model.get_model_id())

        model_id2 = 'id2'
        self.model.model_id = model_id2
        self.assertEqual(model_id2, self.model.model_id)


if __name__ == '__main__':
    unittest.main()
