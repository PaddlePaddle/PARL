#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import numpy as np
import torch
import unittest
from parl.core.torch.policy_distribution import DiagGaussianDistribution, CategoricalDistribution, SoftCategoricalDistribution, SoftMultiCategoricalDistribution


class DiagGaussianDistributionTest(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.num_actions = 2
        self.mean = torch.rand(size=(self.batch_size, self.num_actions))
        self.std = torch.rand(size=(self.batch_size, self.num_actions))
        self.logits = (self.mean, self.std)
        self.dist = DiagGaussianDistribution(self.logits)

    def get_dist(self):
        mean = torch.rand(size=(self.batch_size, self.num_actions))
        std = torch.rand(size=(self.batch_size, self.num_actions))
        logits = (mean, std)
        dist = DiagGaussianDistribution(logits)
        return dist

    def test_sample(self):
        # check shape is [BATCH_SIZE, NUM_ACTIOINS]
        sample_actions = self.dist.sample()
        self.assertTrue(len(sample_actions.shape) == 2 and sample_actions.shape[0] == self.batch_size and \
                        sample_actions.shape[1] == self.num_actions)

    def test_entropy(self):
        # check shape is [BATCHSIZE, ]
        entropy = self.dist.entropy()
        self.assertTrue(entropy.shape == (self.batch_size, ))

    def test_lop(self):
        logp = self.dist.logp(actions=self.dist.sample())
        # check shape is [BATCHSIZE, ]
        self.assertTrue(logp.shape == (self.batch_size, ))
        # range check of logp, the maximum log of probability should be smaller than zero
        self.assertTrue(torch.max(logp) <= 0)

    def test_kl(self):
        # check shape is [BATCHSIZE, ]
        dist2 = self.get_dist()
        kl = self.dist.kl(dist2)
        self.assertTrue(len(kl.shape) == 1 and kl.shape[0] == self.batch_size)

        # kl of the same distribution should be zero
        same_dist = copy.deepcopy(self.dist)
        kl = self.dist.kl(same_dist)
        self.assertTrue(torch.count_nonzero(kl) == 0)

    def test_init_with_wrong_logits_shape(self):
        # input logits with wrong shape
        wrong_logits = torch.rand(size=(self.batch_size, self.num_actions))
        with self.assertRaises(AssertionError):
            DiagGaussianDistribution(wrong_logits)


class CategoricalDistributionTest(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.num_actions = 2
        self.logits = torch.rand(size=(self.batch_size, self.num_actions))
        self.dist = CategoricalDistribution(self.logits)

    def get_dist(self):
        logits = torch.rand(size=(self.batch_size, self.num_actions))
        dist = CategoricalDistribution(logits)
        return dist

    def test_sample(self):
        # check shape
        sample_action = self.dist.sample()
        self.assertEqual(list(sample_action.shape), [
            self.batch_size,
        ])
        # range check
        self.assertGreaterEqual(torch.max(sample_action), 0)
        self.assertLess(torch.min(sample_action), self.num_actions)

    def test_entropy(self):
        # check shape is [BATCHSIZE, ]
        entropy = self.dist.entropy()
        self.assertTrue(
            len(entropy.shape) == 1 and entropy.shape[0] == self.batch_size)

    def test_lop(self):
        sample_action = np.random.choice(
            a=range(self.num_actions), size=self.batch_size)
        sample_action = torch.tensor(sample_action)
        logp = self.dist.logp(sample_action)
        # check shape is [BATCHSIZE, ]
        self.assertEqual(logp.shape, (self.batch_size, self.num_actions))

    def test_kl(self):
        # check shape is [BATCHSIZE, ]
        dist2 = self.get_dist()
        kl = self.dist.kl(dist2)
        self.assertTrue(len(kl.shape) == 1 and kl.shape[0] == self.batch_size)

        # kl of the same distribution should be zero
        same_dist = copy.deepcopy(self.dist)
        kl = self.dist.kl(self.dist)
        ep = 1e-2
        self.assertLessEqual(torch.sum(kl).item(), ep)

    def test_init_with_wrong_logit_shape(self):
        # input logits with wrong shape
        wrong_logits = torch.rand(size=(self.batch_size, 1, self.num_actions))
        with self.assertRaises(AssertionError):
            CategoricalDistribution(wrong_logits)


class SoftCategoricalDistributionTest(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.num_actions = 2
        self.logits = torch.rand(size=(self.batch_size, self.num_actions))
        self.dist = SoftCategoricalDistribution(self.logits)

    def get_dist(self):
        logits = torch.rand(size=(self.batch_size, self.num_actions))
        dist = SoftCategoricalDistribution(logits)
        return dist

    def test_sample(self):
        # check shape
        sample_action = self.dist.sample()
        self.assertTrue(sample_action.shape == (self.batch_size,
                                                self.num_actions))
        # check range of softmax output
        self.assertTrue(
            torch.max(sample_action) <= 1 and torch.min(sample_action) >= 0)

    def test_entropy(self):
        # check shape is [BATCHSIZE, ]
        entropy = self.dist.entropy()
        self.assertTrue(
            len(entropy.shape) == 1 and entropy.shape[0] == self.batch_size)

    def test_init_with_wrong_logit_shape(self):
        # input logits with wrong shape
        wrong_logits = torch.rand(size=(self.batch_size, 1, self.num_actions))
        with self.assertRaises(AssertionError):
            SoftCategoricalDistribution(wrong_logits)


if __name__ == '__main__':
    unittest.main()
