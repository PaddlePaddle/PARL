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

import random
import numpy as np
import paddle
import unittest
import copy
from parl.core.paddle.policy_distribution import DiagGaussianDistribution, CategoricalDistribution, SoftCategoricalDistribution


def set_random_seed(seed: int) -> None:
    """
    Seed the different random generators.
    :param seed:
    :param using_cuda:
    """
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # seed the RNG for all devices (both CPU and CUDA)
    paddle.seed(seed)


class DiagGaussianDistributionTest(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.num_actions = 2
        self.mean = paddle.rand(shape=(self.batch_size, self.num_actions))
        self.log_std = paddle.rand(shape=(self.batch_size, self.num_actions))
        self.logits = (self.mean, self.log_std)
        self.dist = DiagGaussianDistribution(self.logits)

    def paddle_check_eq(self, input, output, rtol=1e-4):
        return paddle.allclose(x=input, y=output, rtol=rtol)

    def get_dist(self, mean=None, log_std=None):
        if mean is None:
            mean = paddle.rand(shape=(self.batch_size, self.num_actions))
        if log_std is None:
            log_std = paddle.rand(shape=(self.batch_size, self.num_actions))
        logits = (mean, log_std)
        dist = DiagGaussianDistribution(logits)
        return dist

    def test_sample(self):
        # check shape is [BATCH_SIZE, NUM_ACTIOINS]
        sample_actions = self.dist.sample()
        self.assertTrue(len(sample_actions.shape) == 2 and sample_actions.shape[0] == self.batch_size and \
                        sample_actions.shape[1] == self.num_actions)

        # test with IO
        mean = paddle.zeros(shape=(self.batch_size, self.num_actions))
        log_std = paddle.ones(shape=(self.batch_size, self.num_actions))
        logits = (mean, log_std)
        dist = DiagGaussianDistribution(logits)
        set_random_seed(12)
        # the standard gaussian variable sampled is fixed when the seed is given
        randn_fixed = paddle.randn(shape=mean.shape)
        set_random_seed(12)
        gaussian_sample = dist.sample()
        fixed_gaussian = randn_fixed * dist.std + dist.mean
        self.assertTrue(self.paddle_check_eq(fixed_gaussian, gaussian_sample))

    def test_entropy(self):
        # check shape is [BATCHSIZE, ]
        entropy = self.dist.entropy()
        self.assertEqual(entropy.shape, [
            self.batch_size,
        ])

        # test with IO
        # when the std is 1/sqrt(2*pi*e) the entropy output expect to be zero
        mean = paddle.rand(shape=(self.batch_size, self.num_actions))
        std = paddle.ones(shape=(self.batch_size, self.num_actions)) * np.sqrt(
            1 / (2 * np.pi * np.e))
        log_std = paddle.log(std)
        dist = self.get_dist(mean, log_std)
        entropy = dist.entropy()
        self.assertTrue(
            self.paddle_check_eq(entropy,
                                 paddle.zeros_like(entropy, dtype='float32')))

    def test_lop(self):
        logp = self.dist.logp(actions=self.dist.sample())
        # check shape is [BATCHSIZE, ]
        self.assertEqual(logp.shape, [
            self.batch_size,
        ])
        # range check of logp, the maximum log of probability should be smaller than zero
        self.assertTrue(paddle.max(logp) <= 0)

        # test with IO
        # when we sample the action that exactly is mean, the logp of each dimension should be (-ln(std)-0.5*ln(2pi))
        logp_output = self.dist.logp(actions=self.dist.mean)
        print(logp_output)
        logp_expect = paddle.sum(
            -self.dist.logstd - 0.5 * np.log(2 * np.pi), axis=-1)
        print(logp_expect)
        self.assertTrue(self.paddle_check_eq(logp_output, logp_expect))

    def test_kl(self):
        # check shape is [BATCHSIZE, ]
        dist2 = self.get_dist()
        kl = self.dist.kl(dist2)
        self.assertTrue(len(kl.shape) == 1 and kl.shape[0] == self.batch_size)

        # kl of the same distribution should be zero
        same_dist = copy.deepcopy(self.dist)
        kl = self.dist.kl(same_dist)
        self.assertTrue(np.count_nonzero(kl) == 0)

        # test with IO
        # two dist that share same mean but different std which satisfy std1 = 0.5 * std2
        # the kl expect to be ln2-(3/8)*num_actions
        mean = paddle.rand(shape=(self.batch_size, self.num_actions))
        std2 = paddle.rand(shape=(self.batch_size, self.num_actions))
        std1 = 0.5 * std2
        dist1 = self.get_dist(mean=mean, log_std=paddle.log(std1))
        dist2 = self.get_dist(mean=mean, log_std=paddle.log(std2))
        kl = dist1.kl(dist2)
        single_kl_expect = np.log(2) - (3 / 8)
        expect_ouput = paddle.ones_like(
            kl) * self.num_actions * single_kl_expect
        self.assertTrue(self.paddle_check_eq(kl, expect_ouput))

    def test_init_with_wrong_logits_shape(self):
        # input logits with wrong shape
        wrong_logits = paddle.rand(shape=(self.batch_size, self.num_actions))
        with self.assertRaises(AssertionError):
            DiagGaussianDistribution(wrong_logits)


class CategoricalDistributionTest(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.num_actions = 3
        self.logits = paddle.rand(shape=(self.batch_size, self.num_actions))
        self.dist = CategoricalDistribution(self.logits)

    def paddle_check_eq(self, input, output, rtol=1e-4):
        return paddle.allclose(x=input, y=output, rtol=rtol)

    def get_dist(self, logits=None):
        if logits is None:
            logits = paddle.rand(shape=(self.batch_size, self.num_actions))
        dist = CategoricalDistribution(logits)
        return dist

    def test_sample(self):
        # check shape
        sample_action = self.dist.sample()
        self.assertEqual(sample_action.shape, [
            self.batch_size,
        ])
        # range check
        self.assertGreaterEqual(paddle.max(sample_action), 0)
        self.assertLess(paddle.min(sample_action), self.num_actions)

        # construct a logit to ouput determined class
        fool_logits = paddle.zeros(shape=(self.batch_size, self.num_actions))
        fool_logits[:, 0] = 9999999999
        set_random_seed(12)
        sample_action = self.get_dist(fool_logits).sample()
        print(sample_action)
        sample_action = paddle.cast(sample_action, 'float32')
        ground_truth = paddle.cast(paddle.zeros_like(sample_action), 'float32')
        self.assertTrue(self.paddle_check_eq(
            sample_action,
            ground_truth))  # expect the sampled action to be zero

    def test_entropy(self):
        # check shape is [BATCHSIZE, ]
        entropy = self.dist.entropy()
        self.assertTrue(
            len(entropy.shape) == 1 and entropy.shape[0] == self.batch_size)

        # uniform distribution should output the maximum entropy, i.e, ln(num_action)
        uniform_logits = paddle.ones(shape=(self.batch_size, self.num_actions))
        entropy_out = self.get_dist(uniform_logits).entropy()
        entropy_exp = np.log(self.num_actions)
        entropy_exp = paddle.ones_like(entropy_out) * entropy_exp
        self.assertTrue(self.paddle_check_eq(entropy_exp, entropy_out))

    def test_lop(self):
        sample_action = np.random.choice(
            a=range(self.num_actions), size=self.batch_size)
        sample_action = paddle.to_tensor(sample_action)
        logp = self.dist.logp(sample_action)
        # check shape is [BATCHSIZE, ]
        self.assertEqual(logp.shape, [
            self.batch_size,
        ])

        # the logp output of the uniform distribution should be log(1/num_actions)
        uniform_logits = paddle.ones(shape=(self.batch_size, self.num_actions))
        dist = self.get_dist(uniform_logits)
        act_smp = dist.sample()
        logp_out = dist.logp(act_smp)
        exp_out = np.log(1 / self.num_actions) + 1e-6
        exp_out = paddle.ones_like(logp_out) * exp_out
        self.assertTrue(self.paddle_check_eq(logp_out, exp_out))

    def test_kl(self):
        # check shape is [BATCHSIZE, ]
        dist2 = self.get_dist()
        kl = self.dist.kl(dist2)
        self.assertTrue(len(kl.shape) == 1 and kl.shape[0] == self.batch_size)

        # kl of the same distribution should be zero
        same_dist = copy.deepcopy(self.dist)
        kl = self.dist.kl(self.dist)
        ep = 1e-2
        self.assertLessEqual(paddle.sum(kl).item(), ep)

    def test_init_with_wrong_logit_shape(self):
        # input logits with wrong shape
        wrong_logits = paddle.rand(
            shape=(self.batch_size, 1, self.num_actions))
        with self.assertRaises(AssertionError):
            CategoricalDistribution(wrong_logits)


class SoftCategoricalDistributionTest(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.num_actions = 2
        self.logits = paddle.rand(shape=(self.batch_size, self.num_actions))
        self.dist = SoftCategoricalDistribution(self.logits)

    def get_dist(self):
        logits = paddle.rand(shape=(self.batch_size, self.num_actions))
        dist = SoftCategoricalDistribution(logits)
        return dist

    def test_sample(self):
        # check shape
        sample_action = self.dist.sample()
        self.assertEqual(sample_action.shape,
                         [self.batch_size, self.num_actions])
        # check range of softmax output
        self.assertTrue(
            paddle.max(sample_action) <= 1 and paddle.min(sample_action) >= 0)

    def test_entropy(self):
        # check shape is [BATCHSIZE, ]
        entropy = self.dist.entropy()
        self.assertTrue(
            len(entropy.shape) == 1 and entropy.shape[0] == self.batch_size)

    def test_init_with_wrong_logit_shape(self):
        # input logits with wrong shape
        wrong_logits = paddle.rand(
            shape=(self.batch_size, 1, self.num_actions))
        with self.assertRaises(AssertionError):
            SoftCategoricalDistribution(wrong_logits)


if __name__ == '__main__':
    unittest.main()
