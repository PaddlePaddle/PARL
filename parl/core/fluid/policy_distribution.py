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

from parl.core.fluid import layers

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
        probs = layers.softmax(self.logits)
        sample_actions = layers.sampling_id(probs)
        return sample_actions

    def entropy(self):
        """
        Returns:
            entropy: A float32 tensor with shape [BATCH_SIZE] of entropy of self policy distribution.
        """
        logits = self.logits - layers.reduce_max(self.logits, dim=1)
        e_logits = layers.exp(logits)
        z = layers.reduce_sum(e_logits, dim=1)
        prob = e_logits / z
        entropy = -1.0 * layers.reduce_sum(
            prob * (logits - layers.log(z)), dim=1)

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

        logits = self.logits - layers.reduce_max(self.logits, dim=1)
        e_logits = layers.exp(logits)
        z = layers.reduce_sum(e_logits, dim=1)
        prob = e_logits / z

        actions = layers.unsqueeze(actions, axes=[1])
        actions_onehot = layers.one_hot(actions, prob.shape[1])
        actions_onehot = layers.cast(actions_onehot, dtype='float32')
        actions_prob = layers.reduce_sum(prob * actions_onehot, dim=1)

        actions_prob = actions_prob + eps
        actions_log_prob = layers.log(actions_prob)

        return actions_log_prob

    def kl(self, other):
        """
        Args:
            other: object of CategoricalDistribution

        Returns:
            kl: A float32 tensor with shape [BATCH_SIZE]
        """
        assert isinstance(other, CategoricalDistribution)

        logits = self.logits - layers.reduce_max(self.logits, dim=1)
        other_logits = other.logits - layers.reduce_max(other.logits, dim=1)

        e_logits = layers.exp(logits)
        other_e_logits = layers.exp(other_logits)

        z = layers.reduce_sum(e_logits, dim=1)
        other_z = layers.reduce_sum(other_e_logits, dim=1)

        prob = e_logits / z
        kl = layers.reduce_sum(
            prob *
            (logits - layers.log(z) - other_logits + layers.log(other_z)),
            dim=1)
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
        logits_shape = layers.cast(layers.shape(self.logits), dtype='int64')
        uniform = layers.uniform_random(logits_shape, min=eps, max=1.0 - eps)
        soft_uniform = layers.log(-1.0 * layers.log(uniform))
        return layers.softmax(self.logits - soft_uniform, axis=-1)


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
                layers.split(
                    input=logits,
                    num_or_sections=list(high - low + 1),
                    dim=len(logits.shape) - 1)))

    def sample(self):
        """
        Returns:
            sample_action: An int64 tensor with shape [BATCH_SIZE, NUM_ACTIOINS] of sample action,
                           with noise to keep the target close to the original action.
        """
        cate_list = []
        for i in range(len(self.categoricals)):
            cate_list.append(self.low[i] + self.categoricals[i].sample())
        return layers.concat(cate_list, axis=-1)

    def layers_add_n(self, input_list):
        """
        Adds all input tensors element-wise, can replace tf.add_n
        """
        assert len(input_list) >= 1
        res = input_list[0]
        for i in range(1, len(input_list)):
            res = layers.elementwise_add(res, input_list[i])
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
