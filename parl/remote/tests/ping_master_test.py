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
import parl
import unittest
from parl.remote.client import disconnect


class TestPingMaster(unittest.TestCase):
    def tearDown(self):
        disconnect()

    def test_throw_exception(self):
        with self.assertRaises(AssertionError):
            parl.connect("176.2.3.4:8080")


if __name__ == '__main__':
    unittest.main()
