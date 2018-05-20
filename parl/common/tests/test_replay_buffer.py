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
from parl.common.replay_buffer import Experience, Sample, ReplayBuffer, ExperienceQueue


class ExperienceForTest(Experience):
    def __init__(self, obs, reward, actions, new_field, status):
        super(ExperienceForTest, self).__init__([obs, reward], [], actions,
                                                status)
        self.new_field = new_field


class TestExperienceQueue(unittest.TestCase):
    def test_single_instance_sampling(self):
        exp_q = ExperienceQueue(sample_seq=False)
        e0 = Experience([np.zeros(10), 1], [], 0, 1)
        e1 = Experience([np.zeros(10), 1], [], 1, 0)
        e2 = Experience([np.zeros(10), 1], [], 2, 0)
        e3 = Experience([np.zeros(10), 1], [], 3, 1)
        e4 = Experience([np.zeros(10), 1], [], 4, 0)
        exp_q.add(e0)
        exp_q.add(e1)
        exp_q.add(e2)
        exp_q.add(e3)
        exp_q.add(e4)
        exp_seqs = exp_q.sample()
        self.assertEqual(len(exp_seqs), 2)
        self.assertEqual(len(exp_q), 1)
        self.assertEqual(exp_seqs[0][0].actions, 1)
        self.assertEqual(exp_seqs[0][0].next_exp.actions, 2)
        self.assertEqual(exp_seqs[1][0].actions, 2)
        self.assertEqual(exp_seqs[1][0].next_exp.actions, 3)
        e5 = Experience([np.zeros(10), 1], [], 5, 1)
        exp_q.add(e5)
        exp_seqs = exp_q.sample()
        self.assertEqual(len(exp_seqs), 1)
        self.assertEqual(len(exp_seqs[0]), 1)
        self.assertEqual(exp_seqs[0][0].actions, 4)
        self.assertEqual(exp_seqs[0][0].next_exp.actions, 5)
        self.assertEqual(len(exp_q), 0)

    def test_sequence_sampling(self):
        exp_q = ExperienceQueue(sample_seq=True)
        e0 = Experience([np.zeros(10), 1], [], 0, 1)
        e1 = Experience([np.zeros(10), 1], [], 1, 0)
        e2 = Experience([np.zeros(10), 1], [], 2, 0)
        e3 = Experience([np.zeros(10), 1], [], 3, 1)
        e4 = Experience([np.zeros(10), 1], [], 4, 0)
        e5 = Experience([np.zeros(10), 1], [], 5, 0)
        exp_q.add(e0)
        exp_q.add(e1)
        exp_q.add(e2)
        exp_q.add(e3)
        exp_q.add(e4)
        exp_q.add(e5)
        exp_seqs = exp_q.sample()
        self.assertEqual(len(exp_seqs), 2)
        self.assertEqual(len(exp_seqs[0]), 2)
        self.assertEqual(exp_seqs[0][0].actions, 1)
        self.assertEqual(exp_seqs[0][1].actions, 2)
        self.assertEqual(exp_seqs[0][1].next_exp.actions, 3)
        self.assertEqual(len(exp_seqs[1]), 1)
        self.assertEqual(exp_seqs[1][0].actions, 4)
        self.assertEqual(exp_seqs[1][0].next_exp.actions, 5)


class TestReplayBuffer(unittest.TestCase):
    def test_single_instance_replay_buffer(self):
        capacity = 30
        episode_len = 4
        buf = ReplayBuffer(capacity, ExperienceForTest)
        total = 0
        expect_total = 0
        for i in xrange(10 * capacity):
            e = ExperienceForTest(
                obs=np.zeros(10),
                reward=i * 0.5,
                actions=i,
                new_field=np.ones(20),
                status=(i + 1) % episode_len == 0)
            buf.add(e)
            # check the circular queue in the buffer
            self.assertTrue(len(buf) == min(i + 1, capacity))
            if (len(buf) < 2):  # need at least two elements
                continue
            # should raise error when trying to pick up the last element
            with self.assertRaises(LastExpError):
                t = Sample(i % capacity, 1)
                buf.get_experiences(t)
            expect_total += len(buf)
            # neither last element nor episode end should be picked up
            for s in buf.sample(len(buf)):
                try:
                    exps = buf.get_experiences(s)
                    total += 1
                except LastExpError as err:
                    self.fail('test_single_instance_replay_buffer raised '
                              'LastExpError: ' + err.message)
        # check the total number of elements added into the buffer
        self.assertTrue(total == expect_total)
        # detect incompatible Experience type
        with self.assertRaises(TypeError):
            e = Experience([np.zeros(10), i * 0.5], [], i, 0)
            buf.add(e)

    def test_deep_copy(self):
        capacity = 5
        buf = ReplayBuffer(capacity, Experience)
        e0 = Experience(
            sensor_inputs=[np.zeros(10), 0],
            states=[],
            actions=0,
            game_status=0)
        e1 = Experience([np.ones(10) * 2, 1], [], 0, 1)
        buf.add(e0)
        e0.sensor_inputs[0] += 1
        buf.add(e0)
        buf.add(e1)
        s = Sample(0, 2)
        exps = buf.get_experiences(s)
        self.assertEqual(np.sum(exps[0].sensor_inputs[0] == 0), 10)
        self.assertEqual(np.sum(exps[1].sensor_inputs[0] == 1), 10)
        self.assertEqual(np.sum(exps[1].next_exp.sensor_inputs[0] == 2), 10)
        exps[0].next_exp.sensor_inputs[0] += 3
        self.assertEqual(np.sum(exps[1].sensor_inputs[0] == 1), 10)
        exps[1].sensor_inputs[0] += 4
        exps = buf.get_experiences(s)
        self.assertEqual(np.sum(exps[0].next_exp.sensor_inputs[0] == 1), 10)


if __name__ == '__main__':
    unittest.main()
