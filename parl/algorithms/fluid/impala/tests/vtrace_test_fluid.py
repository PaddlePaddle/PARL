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
"""Tests for V-trace.

The following code is mainly referenced and copied from:
https://github.com/deepmind/scalable_agent/blob/master/vtrace_test.py
"""

import copy
import numpy as np
import unittest
from parl import layers
from paddle import fluid
from parameterized import parameterized
from parl.algorithms.fluid.impala import vtrace
from parl.utils import get_gpu_count


def _shaped_arange(*shape):
    """Runs np.arange, converts to float and reshapes."""
    return np.arange(np.prod(shape), dtype=np.float32).reshape(*shape)


def _ground_truth_calculation(behaviour_actions_log_probs,
                              target_actions_log_probs, discounts, rewards,
                              values, bootstrap_value, clip_rho_threshold,
                              clip_pg_rho_threshold):
    """Calculates the ground truth for V-trace in Python/Numpy."""
    log_rhos = target_actions_log_probs - behaviour_actions_log_probs

    vs = []
    seq_len = len(discounts)
    rhos = np.exp(log_rhos)
    cs = np.minimum(rhos, 1.0)
    clipped_rhos = rhos
    if clip_rho_threshold:
        clipped_rhos = np.minimum(rhos, clip_rho_threshold)
    clipped_pg_rhos = rhos
    if clip_pg_rho_threshold:
        clipped_pg_rhos = np.minimum(rhos, clip_pg_rho_threshold)

    # This is a very inefficient way to calculate the V-trace ground truth.
    # We calculate it this way because it is close to the mathematical notation of
    # V-trace.
    # v_s = V(x_s)
    #       + \sum^{T-1}_{t=s} \gamma^{t-s}
    #         * \prod_{i=s}^{t-1} c_i
    #         * \rho_t (r_t + \gamma V(x_{t+1}) - V(x_t))
    # Note that when we take the product over c_i, we write `s:t` as the notation
    # of the paper is inclusive of the `t-1`, but Python is exclusive.
    # Also note that np.prod([]) == 1.
    values_t_plus_1 = np.concatenate([values, bootstrap_value[None, :]],
                                     axis=0)
    for s in range(seq_len):
        v_s = np.copy(values[s])  # Very important copy.
        for t in range(s, seq_len):
            v_s += (np.prod(discounts[s:t], axis=0) * np.prod(cs[s:t], axis=0)
                    * clipped_rhos[t] * (rewards[t] + discounts[t] *
                                         values_t_plus_1[t + 1] - values[t]))
        vs.append(v_s)
    vs = np.stack(vs, axis=0)
    pg_advantages = (clipped_pg_rhos * (rewards + discounts * np.concatenate(
        [vs[1:], bootstrap_value[None, :]], axis=0) - values))

    return vtrace.VTraceReturns(vs=vs, pg_advantages=pg_advantages)


class VtraceTest(unittest.TestCase):
    def setUp(self):
        gpu_count = get_gpu_count()
        if gpu_count > 0:
            place = fluid.CUDAPlace(0)
            self.gpu_id = 0
        else:
            place = fluid.CPUPlace()
            self.gpu_id = -1
        self.executor = fluid.Executor(place)

    @parameterized.expand([('Batch1', 1), ('Batch4', 4)])
    def test_from_importance_weights(self, name, batch_size):
        """Tests V-trace against ground truth data calculated in python."""
        seq_len = 5

        # Create log_rhos such that rho will span from near-zero to above the
        # clipping thresholds. In particular, calculate log_rhos in [-2.5, 2.5),
        # so that rho is in approx [0.08, 12.2).
        log_rhos = _shaped_arange(seq_len, batch_size) / (batch_size * seq_len)
        log_rhos = 5 * (log_rhos - 0.5)  # [0.0, 1.0) -> [-2.5, 2.5).
        # Fake behaviour_actions_log_probs, target_actions_log_probs
        target_actions_log_probs = log_rhos + 1.0
        behaviour_actions_log_probs = np.ones(
            shape=log_rhos.shape, dtype='float32')

        values = {
            'behaviour_actions_log_probs':
            behaviour_actions_log_probs,
            'target_actions_log_probs':
            target_actions_log_probs,
            # T, B where B_i: [0.9 / (i+1)] * T
            'discounts':
            np.array([[0.9 / (b + 1) for b in range(batch_size)]
                      for _ in range(seq_len)],
                     dtype=np.float32),
            'rewards':
            _shaped_arange(seq_len, batch_size),
            'values':
            _shaped_arange(seq_len, batch_size) / batch_size,
            'bootstrap_value':
            _shaped_arange(batch_size) + 1.0,
            'clip_rho_threshold':
            3.7,
            'clip_pg_rho_threshold':
            2.2,
        }

        # Calculated by numpy/python
        ground_truth_v = _ground_truth_calculation(**values)

        # Calculated by Fluid
        test_program = fluid.Program()
        with fluid.program_guard(test_program):
            behaviour_actions_log_probs_input = layers.data(
                name='behaviour_actions_log_probs',
                shape=[seq_len, batch_size],
                dtype='float32',
                append_batch_size=False)
            target_actions_log_probs_input = layers.data(
                name='target_actions_log_probs',
                shape=[seq_len, batch_size],
                dtype='float32',
                append_batch_size=False)
            discounts_input = layers.data(
                name='discounts',
                shape=[seq_len, batch_size],
                dtype='float32',
                append_batch_size=False)
            rewards_input = layers.data(
                name='rewards',
                shape=[seq_len, batch_size],
                dtype='float32',
                append_batch_size=False)
            values_input = layers.data(
                name='values',
                shape=[seq_len, batch_size],
                dtype='float32',
                append_batch_size=False)
            bootstrap_value_input = layers.data(
                name='bootstrap_value',
                shape=[batch_size],
                dtype='float32',
                append_batch_size=False)
            fluid_inputs = {
                'behaviour_actions_log_probs':
                behaviour_actions_log_probs_input,
                'target_actions_log_probs': target_actions_log_probs_input,
                'discounts': discounts_input,
                'rewards': rewards_input,
                'values': values_input,
                'bootstrap_value': bootstrap_value_input,
                'clip_rho_threshold': 3.7,
                'clip_pg_rho_threshold': 2.2,
            }
            output = vtrace.from_importance_weights(**fluid_inputs)

        self.executor.run(fluid.default_startup_program())
        feed = copy.copy(values)
        del feed['clip_rho_threshold']
        del feed['clip_pg_rho_threshold']
        [output_vs, output_pg_advantage] = self.executor.run(
            test_program,
            feed=feed,
            fetch_list=[output.vs, output.pg_advantages])

        np.testing.assert_almost_equal(ground_truth_v.vs, output_vs, 5)
        np.testing.assert_almost_equal(ground_truth_v.pg_advantages,
                                       output_pg_advantage, 5)


if __name__ == '__main__':
    unittest.main()
