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


@parl.remote
class Simulator:
    def __init__(self, arg1, arg2=None):
        self.arg1 = arg1
        self.arg2 = arg2

    def get_arg1(self):
        return self.arg1

    def get_arg2(self):
        return self.arg2

    def set_arg1(self, value):
        self.arg1 = value

    def set_arg2(self, value):
        self.arg2 = value


class TestRemoteDecorator(unittest.TestCase):
    def test_instance_in_local(self):
        local_sim = Simulator(1, 2)

        self.assertEqual(local_sim.get_arg1(), 1)
        self.assertEqual(local_sim.get_arg2(), 2)

        local_sim.set_arg1(3)
        local_sim.set_arg2(4)

        self.assertEqual(local_sim.get_arg1(), 3)
        self.assertEqual(local_sim.get_arg2(), 4)

    def test_instance_in_local_with_wrong_getattr_get_variable(self):
        local_sim = Simulator(1, 2)

        try:
            local_sim.get_arg3()
        except AttributeError:
            return

        assert False  # This line should not be executed.

    def test_instance_in_local_with_wrong_getattr_set_variable(self):
        local_sim = Simulator(1, 2)

        try:
            local_sim.set_arg3(3)
        except AttributeError:
            return

        assert False  # This line should not be executed.


if __name__ == '__main__':
    unittest.main()
