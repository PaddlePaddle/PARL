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
from parl.common.error_handling import LastElementError
from parl.common.replay_buffer import Experience, Sample, ReplayBuffer

class ExperienceForTest(Experience):
    def __init__(self, obs, reward, actions, new_field, status):
        super(ExperienceForTest, self).__init__([obs, reward], [], actions, status)
        self.new_field = new_field

class TestReplayBuffer(unittest.TestCase):
    def test_single_instance_replay_buffer(self):
        capacity = 30
        episode_len = 4
        buf = ReplayBuffer(capacity, ExperienceForTest)
        total = 0
        expect_total = 0
        for i in xrange(10 * capacity):
            e = ExperienceForTest(
                np.zeros(10), i*0.5, i, np.ones(20), (i + 1) % episode_len == 0)
            buf.add(e)
            # check the circular queue in the buffer
            self.assertTrue(len(buf) == min(i + 1, capacity))
            if (len(buf) < 2):  # need at least two elements
                continue
            # should raise error when trying to pick up the last element
            with self.assertRaises(LastElementError):
                t = Sample(i % capacity, 1)
                buf.get_experiences(t)
            expect_total += len(buf)
            # neither last element nor episode end should be picked up
            for s in buf.sample(len(buf)):
                try:
                    exps = buf.get_experiences(s)
                    total += 1
                except LastElementError as err:
                    self.fail('test_single_instance_replay_buffer raised '
                              'LastElementError: ' + err.message)
        # check the total number of elements added into the buffer
        self.assertTrue(total == expect_total)
        # detect incompatible Experience type
        with self.assertRaises(TypeError):
            e = Experience([np.zeros(10), i*0.5], [], i, 0)
            buf.add(e)


if __name__ == '__main__':
    unittest.main(exit=False)
