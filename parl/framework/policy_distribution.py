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
        """
        self.dist represents the quantities that characterize the distribution.
        For example, for a Normal distribution, this can be a tuple of (mean, std).
        The actual form of self.dist is defined by the user.
        """
        self.dist = dist

    @abstractmethod
    def __call__(self):
        """
        Implement __call__ to sample an instance.
        """
        pass

    @property
    @abstractmethod
    def dim(self):
        """
        For discrete policies, this function returns the number of actions.
        For continuous policies, this function returns the action vector length.
        For sequential policies (e.g., sentences), this function returns the number
        of choices at each step.
        """
        pass

    def add_uniform_exploration(self, rate):
        """
        Given a uniform exploration rate, this function modifies the distribution.
        The rate could be a floating number of a Variable.
        """
        return NotImplementedError()

    def loglikelihood(self, action):
        """
        Given an action, this function returns the log likelihood of this action under
        the current distribution.
        """
        raise NotImplementedError()


class CategoricalDistribution(PolicyDistribution):
    def __init__(self, dist):
        super(CategoricalDistribution, self).__init__(dist)
        assert isinstance(dist, Variable)

    def __call__(self):
        return comf.categorical_random(self.dist)

    @property
    def dim(self):
        assert len(self.dist.shape) == 2
        return self.dist.shape[1]

    def add_uniform_exploration(self, rate):
        if not (isinstance(rate, float) and rate == 0):
            self.dist = self.dist * (1 - rate) + \
                   1 / float(self.dim) * rate

    def loglikelihood(self, action):
        assert isinstance(action, Variable)
        assert action.dtype == convert_np_dtype_to_dtype_("int") \
            or action.dtype == convert_np_dtype_to_dtype_("int64")
        return 0 - layers.cross_entropy(input=self.dist, label=action)


class Deterministic(PolicyDistribution):
    def __init__(self, dist):
        super(Deterministic, self).__init__(dist)
        ## For deterministic action, we only support continuous ones
        assert isinstance(dist, Variable)
        assert dist.dtype == convert_np_dtype_to_dtype_("float32") \
            or dist.dtype == convert_np_dtype_to_dtype_("float64")

    @property
    def dim(self):
        assert len(self.dist.shape) == 2
        return self.dist.shape[1]

    def __call__(self):
        return self.dist


def q_categorical_distribution(q_value):
    """
    Generate a PolicyDistribution object given a Q value.
    We construct a one-hot distribution according to the Q value.
    """
    assert len(q_value.shape) == 2, "[batch_size, num_actions]"
    max_id = comf.argmax_layer(q_value)
    prob = layers.cast(
        x=layers.one_hot(
            input=max_id, depth=q_value.shape[-1]),
        dtype="float32")
    return CategoricalDistribution(prob)
