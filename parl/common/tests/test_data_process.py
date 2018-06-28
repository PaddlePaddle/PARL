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
from parl.common.data_process import DataSpecs, DataProcessor
from parl.common.replay_buffer import Experience


class TestDataProcess(unittest.TestCase):
    def testd_DataSpecs(self):
        """
        Test DataSpecs in the following cases:
            - no `dtype` specified
            - `shape=[]`
            - empty specs for some types, e.g., `rewards = []`
        """
        specs = DataSpecs(
            inputs=[("sensor", dict(
                shape=[10], dtype="uint8")), ("sensor2", dict(shape=[5]))],
            states=[("state", dict(shape=[20]))],
            actions=[("action", dict(shape=[]))],
            rewards=[],
            next_episode_end=[("next_episode_end", dict(
                shape=[1], dtype="int32"))])
        self.assertEqual(
            specs.inputs, [("sensor", dict(
                shape=[10], dtype="uint8")), ("sensor2", dict(
                    shape=[5], dtype="float32"))])
        self.assertEqual(
            specs.next_inputs, [("next_sensor", dict(
                shape=[10], dtype="uint8")), ("next_sensor2", dict(
                    shape=[5], dtype="float32"))])
        self.assertEqual(
            specs.states, [("state", dict(
                shape=[20], dtype="float32"))])
        self.assertEqual(
            specs.next_states, [("next_state", dict(
                shape=[20], dtype="float32"))])
        self.assertEqual(
            specs.actions, [("action", dict(
                shape=[1], dtype="float32"))])
        self.assertEqual(specs.rewards, [])
        self.assertEqual(
            specs.next_episode_end, [("next_episode_end", dict(
                shape=[1], dtype="int32"))])

    def test_learning_process(self):
        """
        Test if `DataProcessor.process_learning_data' handles correctly:
            - `states` data
            - `next_*` data
            - data with variable shapes, i.e., `action`
            - unused data of `Experience`, .e., `reward`
        """
        specs = DataSpecs(
            inputs=[("sensor", dict(
                shape=[10], dtype="uint8")), ("sensor2", dict(shape=[5]))],
            states=[("state", dict(shape=[20]))],
            actions=[("action", dict(shape=[]))],
            rewards=[],
            next_episode_end=[("next_episode_end", dict(
                shape=[1], dtype="int32"))])

        data_proc = DataProcessor(specs)

        # `exp_cls` has `reward` data that `specs` does not use
        exp_cls = Experience.define(
            "Experience",
            ["sensor", "sensor2", "state", "action", "reward", "episode_end"])

        e0 = exp_cls(
            sensor=np.random.randint(0, 255, 10),
            sensor2=np.random.randint(0, 255, 5),
            state=np.random.rand(20),
            action=[1],
            reward=np.array([0]),
            episode_end=np.array([0]).astype('uint8'))
        e1 = exp_cls(
            sensor=np.random.randint(0, 255, 10),
            sensor2=np.random.randint(0, 255, 5),
            state=np.random.rand(20),
            action=[2],
            reward=np.array([1]),
            episode_end=np.array([0]).astype('uint8'))
        e2 = exp_cls(
            sensor=np.random.randint(0, 255, 10),
            sensor2=np.random.randint(0, 255, 5),
            state=np.random.rand(20),
            action=[3],
            reward=np.array([2]),
            episode_end=np.array([1]).astype('uint8'))
        e3 = exp_cls(
            sensor=np.random.randint(0, 255, 10),
            sensor2=np.random.randint(0, 255, 5),
            state=np.random.rand(20),
            action=[4],
            reward=np.array([3]),
            episode_end=np.array([0]).astype('uint8'))
        e4 = exp_cls(
            sensor=np.random.randint(0, 255, 10),
            sensor2=np.random.randint(0, 255, 5),
            state=np.random.rand(20),
            action=[5],
            reward=np.array([4]),
            episode_end=np.array([0]).astype('uint8'))
        e0.next_exp = e1
        e1.next_exp = e2
        e2.next_exp = e3
        e3.next_exp = e4
        # Two exp. sequences, [e0, e1, e2] and [e3, e4]
        exp_seqs = [[e0, e1], [e3]]

        data = data_proc.process_learning_data(exp_seqs)
        self.assertIn("rewards", data)
        self.assertEqual(data["rewards"], dict())
        for k, v in vars(specs).iteritems():
            self.assertIn(k, data)
            for spec in v:
                self.assertIn(spec[0], data[k])
                # there should be no `reward` in `data
                self.assertNotEqual(spec[0], "reward")
                d = data[k][spec[0]]
                self.assertEqual(type(d), np.ndarray)
                if spec[0] == "next_episode_end":
                    self.assertEqual(d.shape, (3, 1))
                    self.assertEqual(d[0, :], e1.episode_end)
                    self.assertEqual(d[1, :], e2.episode_end)
                    self.assertEqual(d[2, :], e4.episode_end)
                elif spec[0] == "action":
                    self.assertEqual(d.shape, (3, 1))
                    self.assertTrue(np.array_equal(d[0], e0.action))
                    self.assertTrue(np.array_equal(d[1], e1.action))
                    self.assertTrue(np.array_equal(d[2], e3.action))
                elif spec[0] == "state":
                    self.assertEqual(d.shape, (2, 20))
                    # For `states` data, we only extract the initial state 
                    # for each sequence.
                    # We don't specify the dtype for 'state`, so by default
                    # it is `float32`
                    self.assertTrue(
                        np.array_equal(d[0], e0.state.astype("float32")))
                    self.assertTrue(
                        np.array_equal(d[1], e3.state.astype("float32")))
                elif spec[0] == "sensor":
                    self.assertEqual(d.shape, (3, 10))
                    self.assertTrue(np.array_equal(d[0], e0.sensor))
                    self.assertTrue(np.array_equal(d[1], e1.sensor))
                    self.assertTrue(np.array_equal(d[2], e3.sensor))
                elif spec[0] == "next_sensor2":
                    self.assertEqual(d.shape, (3, 5))
                    self.assertTrue(
                        np.array_equal(d[0], e1.sensor2.astype("float32")))
                    self.assertTrue(
                        np.array_equal(d[1], e2.sensor2.astype("float32")))
                    self.assertTrue(
                        np.array_equal(d[2], e4.sensor2.astype("float32")))


if __name__ == '__main__':
    unittest.main()
