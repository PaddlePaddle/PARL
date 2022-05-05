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

import paddle
import paddle.nn.functional as F
import numpy as np

__all__ = [
    'PolicyDistribution', 'CategoricalDistribution',
    'SoftCategoricalDistribution', 'SoftMultiCategoricalDistribution'
]


class PolicyDistribution(object):
    def sample(self):
        """Sampling from the policy distribution."""
        raise NotImplementedError

    def entropy(self):
        """The entropy of the policy distribution."""
        raise NotImplementedError

    def kl(self, other):
        """The KL-divergence between self policy distributions and other."""
        raise NotImplementedError

    def logp(self, actions):
        """The log-probabilities of the actions in this policy distribution."""
        raise NotImplementedError


class DiagGaussianDistribution(PolicyDistribution):
    """DiagGaussian distribution for continuous action spaces."""

    def __init__(self, logits):
        """
        Args:
            logits: A tuple of (mean, logstd)
                    mean: A float32 tensor with shape [BATCH_SIZE, NUM_ACTIONS] of unnormalized policy logits
                    logstd: A float32 tensor with shape [BATCH_SIZE, NUM_ACTIONS]
        """
        assert len(logits) == 2
        assert len(logits[0].shape) == 2 and len(logits[1].shape) == 2
        self.logits = logits
        (mean, logstd) = logits
        self.mean = mean
        self.logstd = logstd

        self.std = paddle.exp(self.logstd)

    def sample(self):
        """
        Returns:
            sample_action: An float32 tensor with shape [BATCH_SIZE, NUM_ACTIOINS] of sample action,
                           with noise to keep the target close to the original action.
        """
        mean_shape = paddle.to_tensor(self.mean.shape, dtype='int64')
        random_normal = paddle.normal(shape=mean_shape)
        return self.mean + self.std * random_normal

    def entropy(self):
        """
        Returns:
            entropy: A float32 tensor with shape [BATCH_SIZE] of entropy of self policy distribution.
        """
        entropy = paddle.sum(
            self.logstd + 0.5 * np.log(2.0 * np.pi * np.e), axis=1)
        return entropy

    def logp(self, actions):
        """
        Args:
            actions: An float32 tensor with shape [BATCH_SIZE, NUM_ACTIOINS]

        Returns:
            actions_log_prob: A float32 tensor with shape [BATCH_SIZE]
        """
        assert len(actions.shape) == 2

        norm_actions = paddle.sum(
            paddle.square((actions - self.mean) / self.std), axis=1)
        pi_item = 0.5 * np.log(2.0 * np.pi) * actions.shape[1]
        actions_log_prob = -0.5 * norm_actions - pi_item - paddle.sum(
            self.logstd, axis=1)

        return actions_log_prob

    def kl(self, other):
        """
        Args:
            other: object of DiagGaussianDistribution

        Returns:
            kl: A float32 tensor with shape [BATCH_SIZE]
        """
        assert isinstance(other, DiagGaussianDistribution)

        temp = (paddle.square(self.std) + paddle.square(self.mean - other.mean)
                ) / (2.0 * paddle.square(other.std))
        kl = paddle.sum(other.logstd - self.logstd + temp - 0.5, axis=1)
        return kl


class CategoricalDistribution(PolicyDistribution):
    """Categorical distribution for discrete action spaces."""

    def __init__(self, logits):
        """
        Args:
            logits: A float32 tensor with shape [BATCH_SIZE, NUM_ACTIONS] of unnormalized policy logits
        """
        assert len(logits.shape) == 2
        self.logits = logits

    def sample(self):
        """
        Returns:
            sample_action: An int64 tensor with shape [BATCH_SIZE] of multinomial sampling ids.
                           Each value in sample_action is in [0, NUM_ACTIOINS - 1]
        """
        probs = F.softmax(self.logits)
        sample_actions = paddle.fluid.layers.sampling_id(probs)
        return sample_actions

    def entropy(self):
        """
        Returns:
            entropy: A float32 tensor with shape [BATCH_SIZE] of entropy of self policy distribution.
        """
        logits = self.logits - paddle.max(self.logits, axis=1, keepdim=True)
        e_logits = paddle.exp(logits)
        z = paddle.sum(e_logits, axis=1, keepdim=True)
        prob = e_logits / z
        entropy = -1.0 * paddle.sum(prob * (logits - paddle.log(z)), axis=1)

        return entropy

    def logp(self, actions, eps=1e-6):
        """
        Args:
            actions: An int64 tensor with shape [BATCH_SIZE]
            eps: A small float constant that avoids underflows when computing the log probability

        Returns:
            actions_log_prob: A float32 tensor with shape [BATCH_SIZE]
        """
        assert len(actions.shape) == 1

        logits = self.logits - paddle.max(self.logits, axis=1, keepdim=True)
        e_logits = paddle.exp(logits)
        z = paddle.sum(e_logits, axis=1, keepdim=True)
        prob = e_logits / z

        actions_onehot = F.one_hot(actions, prob.shape[1])
        actions_onehot = paddle.cast(actions_onehot, dtype='float32')
        actions_prob = prob * actions_onehot
        actions_prob = paddle.max(actions_prob, axis=1)

        actions_prob = actions_prob + eps
        actions_log_prob = paddle.log(actions_prob)

        return actions_log_prob

    def kl(self, other):
        """
        Args:
            other: object of CategoricalDistribution

        Returns:
            kl: A float32 tensor with shape [BATCH_SIZE]
        """
        assert isinstance(other, CategoricalDistribution)

        logits = self.logits - paddle.max(self.logits, axis=1, keepdim=True)
        other_logits = other.logits - paddle.max(
            other.logits, axis=1, keepdim=True)

        e_logits = paddle.exp(logits)
        other_e_logits = paddle.exp(other_logits)

        z = paddle.sum(e_logits, axis=1, keepdim=True)
        other_z = paddle.sum(other_e_logits, axis=1, keepdim=True)

        prob = e_logits / z
        kl = paddle.sum(
            prob *
            (logits - paddle.log(z) - other_logits + paddle.log(other_z)),
            axis=1)
        return kl


class SoftCategoricalDistribution(CategoricalDistribution):
    """Categorical distribution with noise for discrete action spaces"""

    def __init__(self, logits):
        """
        Args:
            logits: A float32 tensor with shape [BATCH_SIZE, NUM_ACTIONS] of unnormalized policy logits
        """
        self.logits = logits
        super(SoftCategoricalDistribution, self).__init__(logits)

    def sample(self):
        """
        Returns:
            sample_action: An int64 tensor with shape [BATCH_SIZE, NUM_ACTIOINS] of sample action,
                           with noise to keep the target close to the original action.
        """
        eps = 1e-4
        logits_shape = paddle.to_tensor(self.logits.shape, dtype='int64')
        uniform = paddle.uniform(logits_shape, min=eps, max=1.0 - eps)
        soft_uniform = paddle.log(-1.0 * paddle.log(uniform))
        return F.softmax(self.logits - soft_uniform, axis=-1)


class SoftMultiCategoricalDistribution(PolicyDistribution):
    """Categorical distribution with noise for MultiDiscrete action spaces."""

    def __init__(self, logits, low, high):
        """
        Args:
            logits: A float32 tensor with shape [BATCH_SIZE, LEN_MultiDiscrete, NUM_ACTIONS] of unnormalized policy logits
            low: lower bounds of sample action
            high: Upper bounds of action
        """
        self.logits = logits
        self.low = low
        self.high = high
        self.categoricals = list(
            map(
                SoftCategoricalDistribution,
                paddle.split(
                    logits,
                    num_or_sections=list(high - low + 1),
                    axis=len(logits.shape) - 1)))

    def sample(self):
        """
        Returns:
            sample_action: An int64 tensor with shape [BATCH_SIZE, NUM_ACTIOINS] of sample action,
                           with noise to keep the target close to the original action.
        """
        cate_list = []
        for i in range(len(self.categoricals)):
            cate_list.append(self.low[i] + self.categoricals[i].sample())
        return paddle.concat(cate_list, axis=-1)

    def layers_add_n(self, input_list):
        """
        Adds all input tensors element-wise, can replace tf.add_n
        """
        assert len(input_list) >= 1
        res = input_list[0]
        for i in range(1, len(input_list)):
            res = paddle.add(res, input_list[i])
        return res

    def entropy(self):
        """
        Returns:
            entropy: A float32 tensor with shape [BATCH_SIZE] of entropy of self policy distribution.
        """
        return self.layers_add_n([p.entropy() for p in self.categoricals])

    def kl(self, other):
        """
        Args:
            other: object of SoftCategoricalDistribution

        Returns:
            kl: A float32 tensor with shape [BATCH_SIZE]
        """
        return self.layers_add_n(
            [p.kl(q) for p, q in zip(self.categoricals, other.categoricals)])
