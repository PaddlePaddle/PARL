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

import parl.layers as layers
from paddle.fluid.framework import Variable
from parl.layers import common_functions as comf
from paddle.fluid.framework import convert_np_dtype_to_dtype_
from abc import ABCMeta, abstractmethod


class PolicyDistribution(object):
    __metaclass__ = ABCMeta

    def __init__(self, dist):
        assert len(dist.shape) == 2
        self.dim = dist.shape[1]
        self.dist = dist

    @abstractmethod
    def __call__(self):
        """
        Implement __call__ to sample an instance.
        """
        pass

    def dim(self):
        """
        For discrete policies, this function returns the number of actions.
        For continuous policies, this function returns the action vector length.
        For sequential policies (e.g., sentences), this function returns the number
        of choices at each step.
        """
        return self.dim

    def dist(self):
        return self.dist

    def loglikelihood(self, action):
        """
        Given an action, this function returns the log likelihood of this action under
        the current distribution.
        """
        raise NotImplementedError()


class CategoricalDistribution(PolicyDistribution):
    def __init__(self, dist):
        super(CategoricalDistribution, self).__init__(dist)

    def __call__(self):
        return comf.categorical_random(self.dist)

    def loglikelihood(self, action):
        return 0 - layers.cross_entropy(input=self.dist, label=action)


class Deterministic(PolicyDistribution):
    def __init__(self, dist):
        super(Deterministic, self).__init__(dist)
        ## For deterministic action, we only support continuous ones
        assert dist.dtype == convert_np_dtype_to_dtype_("float32") \
            or dist.dtype == convert_np_dtype_to_dtype_("float64")

    def __call__(self):
        return self.dist

    def loglikelihood(self, action):
        assert False, "You cannot compute likelihood for a deterministic action!"


def q_categorical_distribution(q_value, exploration_rate=0.0):
    """
    Generate a PolicyDistribution object given a Q value.
    We first construct a one-hot distribution according to the Q value,
    and then add an exploration rate to get a probability.
    """
    assert len(q_value.shape) == 2, "[batch_size, num_actions]"
    max_id = comf.argmax_layer(q_value)
    prob = layers.cast(
        x=layers.one_hot(
            input=max_id, depth=q_value.shape[-1]),
        dtype="float32")
    ### exploration_rate could be a Variable
    if not (isinstance(exploration_rate, float) and exploration_rate == 0):
        prob = exploration_rate / float(q_value.shape[-1]) \
               + (1 - exploration_rate) * prob
    return CategoricalDistribution(prob)
