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
from paddle import fluid
from parl import layers
from parameterized import parameterized
from parl.core.fluid.policy_distribution import *
from parl.utils import get_gpu_count, np_softmax, np_cross_entropy


class PolicyDistributionTest(unittest.TestCase):
    def setUp(self):

        gpu_count = get_gpu_count()
        if gpu_count > 0:
            place = fluid.CUDAPlace(0)
            self.gpu_id = 0
        else:
            place = fluid.CPUPlace()
            self.gpu_id = -1
        self.executor = fluid.Executor(place)

    @parameterized.expand([('Batch1', 1), ('Batch5', 5)])
    def test_categorical_distribution(self, name, batch_size):
        ACTIONS_NUM = 4
        test_program = fluid.Program()
        with fluid.program_guard(test_program):
            logits = layers.data(
                name='logits', shape=[ACTIONS_NUM], dtype='float32')
            other_logits = layers.data(
                name='other_logits', shape=[ACTIONS_NUM], dtype='float32')
            actions = layers.data(name='actions', shape=[], dtype='int64')

            categorical_distribution = CategoricalDistribution(logits)
            other_categorical_distribution = CategoricalDistribution(
                other_logits)

            sample_actions = categorical_distribution.sample()
            entropy = categorical_distribution.entropy()
            actions_logp = categorical_distribution.logp(actions)
            kl = categorical_distribution.kl(other_categorical_distribution)

        self.executor.run(fluid.default_startup_program())

        logits_np = np.random.randn(batch_size, ACTIONS_NUM).astype('float32')
        other_logits_np = np.random.randn(batch_size,
                                          ACTIONS_NUM).astype('float32')
        actions_np = np.random.randint(
            0, high=ACTIONS_NUM, size=(batch_size, 1), dtype='int64')

        # ground truth calculated by numpy/python
        gt_probs = np_softmax(logits_np)
        gt_other_probs = np_softmax(other_logits_np)
        gt_log_probs = np.log(gt_probs)
        gt_entropy = -1.0 * np.sum(gt_probs * gt_log_probs, axis=1)

        gt_actions_logp = -1.0 * np_cross_entropy(gt_probs + 1e-6, actions_np)
        gt_actions_logp = np.squeeze(gt_actions_logp, -1)
        gt_kl = np.sum(
            np.where(gt_probs != 0,
                     gt_probs * np.log(gt_probs / gt_other_probs), 0),
            axis=-1)

        # result calculated by CategoricalDistribution
        [
            output_sample_actions, output_entropy, output_actions_logp,
            output_kl
        ] = self.executor.run(
            program=test_program,
            feed={
                'logits': logits_np,
                'other_logits': other_logits_np,
                'actions': np.squeeze(actions_np, axis=1)
            },
            fetch_list=[sample_actions, entropy, actions_logp, kl])

        # test entropy
        np.testing.assert_almost_equal(output_entropy, gt_entropy, 5)

        # test logp
        np.testing.assert_almost_equal(output_actions_logp, gt_actions_logp, 5)

        # test sample
        action_ids = np.arange(ACTIONS_NUM)
        assert np.isin(output_sample_actions, action_ids).all()

        # test kl
        np.testing.assert_almost_equal(output_kl, gt_kl, 5)


if __name__ == '__main__':
    unittest.main()
