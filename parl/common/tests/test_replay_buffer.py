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
from parl.common.replay_buffer import Experience
from parl.common.replay_buffer import ReplayBuffer, NoReplacementQueue, Sample


class TestNoReplacementQueue(unittest.TestCase):
    def test_single_instance_sampling(self):
        exp_q = NoReplacementQueue(sample_seq=False)
        exp_q.add(obs=np.zeros(10),
                  reward=1,
                  state=[],
                  action=0,
                  episode_end=1)
        exp_q.add(obs=np.zeros(10),
                  reward=1,
                  state=[],
                  action=1,
                  episode_end=0)
        exp_q.add(obs=np.zeros(10),
                  reward=1,
                  state=[],
                  action=2,
                  episode_end=0)
        exp_q.add(obs=np.zeros(10),
                  reward=1,
                  state=[],
                  action=3,
                  episode_end=1)
        exp_q.add(obs=np.zeros(10),
                  reward=1,
                  state=[],
                  action=4,
                  episode_end=0)
        exp_seqs = exp_q.sample()
        self.assertEqual(len(exp_seqs), 2)
        self.assertEqual(len(exp_q), 1)
        self.assertEqual(exp_seqs[0][0].action, 1)
        self.assertEqual(exp_seqs[0][0].next_exp.action, 2)
        self.assertEqual(exp_seqs[1][0].action, 2)
        self.assertEqual(exp_seqs[1][0].next_exp.action, 3)
        exp_q.add(obs=np.zeros(10),
                  reward=1,
                  state=[],
                  action=5,
                  episode_end=1)
        exp_seqs = exp_q.sample()
        self.assertEqual(len(exp_seqs), 1)
        self.assertEqual(len(exp_seqs[0]), 1)
        self.assertEqual(exp_seqs[0][0].action, 4)
        self.assertEqual(exp_seqs[0][0].next_exp.action, 5)
        self.assertEqual(len(exp_q), 0)

    def test_sequence_sampling(self):
        exp_q = NoReplacementQueue(sample_seq=True)
        exp_q.add(obs=np.zeros(10),
                  reward=1,
                  state=[],
                  action=0,
                  episode_end=1)
        exp_q.add(obs=np.zeros(10),
                  reward=1,
                  state=[],
                  action=1,
                  episode_end=0)
        exp_q.add(obs=np.zeros(10),
                  reward=1,
                  state=[],
                  action=2,
                  episode_end=0)
        exp_q.add(obs=np.zeros(10),
                  reward=1,
                  state=[],
                  action=3,
                  episode_end=1)
        exp_q.add(obs=np.zeros(10),
                  reward=1,
                  state=[],
                  action=4,
                  episode_end=0)
        exp_q.add(obs=np.zeros(10),
                  reward=1,
                  state=[],
                  action=5,
                  episode_end=0)
        exp_seqs = exp_q.sample()
        self.assertEqual(len(exp_seqs), 2)
        self.assertEqual(len(exp_seqs[0]), 2)
        self.assertEqual(exp_seqs[0][0].action, 1)
        self.assertEqual(exp_seqs[0][1].action, 2)
        self.assertEqual(exp_seqs[0][1].next_exp.action, 3)
        self.assertEqual(len(exp_seqs[1]), 1)
        self.assertEqual(exp_seqs[1][0].action, 4)
        self.assertEqual(exp_seqs[1][0].next_exp.action, 5)


class TestReplayBuffer(unittest.TestCase):
    def test_single_instance_replay_buffer(self):
        capacity = 30
        episode_len = 4
        buf = ReplayBuffer(capacity)
        total = 0
        expect_total = 0
        exp_cls = Experience.define(
            "Experience", ['obs', 'reward', 'action', 'episode_end'])
        for i in xrange(10 * capacity):
            e = exp_cls(
                obs=np.zeros(10),
                reward=i * 0.5,
                action=i,
                episode_end=(i + 1) % episode_len == 0)
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
            e = exp_cls(
                obs=np.zeros(10),
                reward=i * 0.5,
                states=[],
                action=i,
                episode_end=0)
            buf.add(e)

    def test_deep_copy(self):
        capacity = 5
        buf = ReplayBuffer(capacity)
        exp_cls = Experience.define(
            "Experience", ['obs', 'reward', 'action', 'episode_end'])
        e0 = exp_cls(obs=np.zeros(10), reward=0, action=0, episode_end=0)
        e1 = exp_cls(obs=np.ones(10) * 2, reward=1, action=0, episode_end=1)
        buf.add(e0)
        e0.obs += 1
        buf.add(e0)
        buf.add(e1)
        s = Sample(0, 2)
        exps = buf.get_experiences(s)
        self.assertEqual(np.sum(exps[0].obs == 0), 10)
        self.assertEqual(np.sum(exps[1].obs == 1), 10)
        self.assertEqual(np.sum(exps[1].next_exp.obs == 2), 10)
        exps[0].next_exp.obs += 3
        self.assertEqual(np.sum(exps[1].obs == 1), 10)
        exps[1].obs += 4
        exps = buf.get_experiences(s)
        self.assertEqual(np.sum(exps[0].next_exp.obs == 1), 10)


if __name__ == '__main__':
    unittest.main()
