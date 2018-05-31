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
from parl.agent_zoo.agent_helpers import RLDataProcessor
from parl.common.replay_buffer import Experience


class TestRLDataProcessor(unittest.TestCase):
    def test_learning_process(self):
        specs = {
            "inputs": ["sensors", "states"],
            "next_inputs": ["next_sensors", "next_states"],
            "actions": ["actions"],
            "rewards": ["rewards"],
            "next_episode_end": ["next_episode_end"]
        }
        data_proc = RLDataProcessor(specs)

        exp_cls = Experience.define(
            "Experience",
            ["sensors", "states", "actions", "rewards", "episode_end"])

        e0 = exp_cls(
            sensors=np.random.rand(10),
            states=np.random.rand(20),
            actions=[1, 2, 3],
            rewards=np.array([0]),
            episode_end=np.array([0]).astype('uint8'))
        e1 = exp_cls(
            sensors=np.random.rand(10),
            states=np.random.rand(20),
            actions=[4, 5],
            rewards=np.array([1]),
            episode_end=np.array([0]).astype('uint8'))
        e2 = exp_cls(
            sensors=np.random.rand(10),
            states=np.random.rand(20),
            actions=[7, 8, 9, 10],
            rewards=np.array([2]),
            episode_end=np.array([1]).astype('uint8'))
        e3 = exp_cls(
            sensors=np.random.rand(10),
            states=np.random.rand(20),
            actions=[11],
            rewards=np.array([3]),
            episode_end=np.array([0]).astype('uint8'))
        e4 = exp_cls(
            sensors=np.random.rand(10),
            states=np.random.rand(20),
            actions=[12, 13],
            rewards=np.array([4]),
            episode_end=np.array([0]).astype('uint8'))
        e0.next_exp = e1
        e1.next_exp = e2
        e2.next_exp = e3
        e3.next_exp = e4
        exp_seqs = [[e0, e1], [e3]]

        data = data_proc.process_learning_inputs(exp_seqs)
        print data["actions"]
        for k in specs:
            self.assertTrue(k in data)
            for attr in specs[k]:
                self.assertTrue(attr in data[k])
                d = data[k][attr]
                if attr == "actions":
                    self.assertEqual(type(d), list)
                    self.assertEqual(len(d), 3)
                    self.assertEqual(d[0], e0.actions)
                    self.assertEqual(d[1], e1.actions)
                    self.assertEqual(d[2], e3.actions)
                else:
                    self.assertEqual(type(d), np.ndarray)
                    if attr == "rewards":
                        self.assertEqual(d.shape, (3, 1))
                        self.assertEqual(d[0, :], e0.rewards)
                        self.assertEqual(d[1, :], e1.rewards)
                        self.assertEqual(d[2, :], e3.rewards)
                    elif attr == "next_episode_end":
                        self.assertEqual(d.shape, (3, 1))
                        self.assertEqual(d[0, :], e1.episode_end)
                        self.assertEqual(d[1, :], e2.episode_end)
                        self.assertEqual(d[2, :], e4.episode_end)
                    elif attr == "states":
                        self.assertEqual(d.shape, (3, 20))
                        self.assertTrue(np.array_equal(d[0], e0.states))
                        self.assertTrue(np.array_equal(d[1], e1.states))
                        self.assertTrue(np.array_equal(d[2], e3.states))
                    elif attr == "sensors":
                        self.assertEqual(d.shape, (3, 10))
                        self.assertTrue(np.array_equal(d[0], e0.sensors))
                        self.assertTrue(np.array_equal(d[1], e1.sensors))
                        self.assertTrue(np.array_equal(d[2], e3.sensors))
                    elif attr == "next_sensors":
                        self.assertEqual(d.shape, (3, 10))
                        self.assertTrue(np.array_equal(d[0], e1.sensors))
                        self.assertTrue(np.array_equal(d[1], e2.sensors))
                        self.assertTrue(np.array_equal(d[2], e4.sensors))
                    elif attr == "next_states":
                        self.assertEqual(d.shape, (3, 20))
                        self.assertTrue(np.array_equal(d[0], e1.states))
                        self.assertTrue(np.array_equal(d[1], e2.states))
                        self.assertTrue(np.array_equal(d[2], e4.states))


if __name__ == '__main__':
    unittest.main()
