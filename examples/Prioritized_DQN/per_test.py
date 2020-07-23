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

import functools
import random
import time
import unittest
import numpy as np

from per.proportional import ProportionalPER

MEMORY_SIZE = int(1e5)
BATCH_SIZE = 32


def _timeit(func):
    @functools.wraps(func)
    def timeit(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        used_time = time.time() - start_time
        print("Run function `{}` finished, used time: {:.4f}s".format(
            func.__name__, used_time))
        return result

    return timeit


def transition_generator():
    while True:
        obs = np.random.random((1, 4))
        act = np.random.randint(0, 10)
        reward = np.random.randint(0, 2)
        next_obs = np.random.random((1, 4))
        terminal = np.random.choice([True, False])
        transition = (obs, act, reward, next_obs, terminal)
        yield transition


class TestPER(unittest.TestCase):
    def setUp(self):
        self.transition_list = []
        self.trans_gen = transition_generator()
        for _ in range(MEMORY_SIZE):
            transition = next(self.trans_gen)
            self.transition_list.append(transition)

    def _run_op(self, per):
        mid = MEMORY_SIZE >> 1
        for transition in self.transition_list[:mid]:
            per.store(transition)

        # Test sampling from a PER that is not full
        try:
            per.sample()
        except AssertionError:
            pass

        for transition in self.transition_list[mid:]:
            per.store(transition)

        batch, idxs, probs = per.sample()
        self.assertEqual(len(batch), BATCH_SIZE)
        self.assertEqual(len(idxs), BATCH_SIZE)
        self.assertEqual(len(probs), BATCH_SIZE)

        # Test update priorities
        for i in range(100):
            _, idxs, _ = per.sample()
            priors = np.random.random((len(idxs), ))
            per.update(idxs, priors)

        # Test insert into a full PER
        for _ in range(1000):
            transition = next(self.trans_gen)
            per.store(transition)

    @_timeit
    def test_proportional(self):
        per = ProportionalPER(alpha=0, seg_num=BATCH_SIZE, size=MEMORY_SIZE)
        self._run_op(per)

        per = ProportionalPER(alpha=1, seg_num=BATCH_SIZE, size=MEMORY_SIZE)
        self._run_op(per)

        per = ProportionalPER(
            alpha=0.7,
            seg_num=BATCH_SIZE,
            size=MEMORY_SIZE,
            init_mem=self.transition_list)
        self._run_op(per)


if __name__ == "__main__":
    unittest.main()
