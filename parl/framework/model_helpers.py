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


class PolicyDist(object):
    def __init__(self, dist):
        assert len(dist.shape) == 2
        self.dim = dist.shape[1]
        self.dist = dist

    def __call__(self):
        raise NotImplementedError("Implement __call__ to sample an instance!")


class DiscreteDist(PolicyDist):
    def __init__(self, dist):
        super(DiscreteDist, self).__init__(dist)

    def __call__(self):
        return comf.discrete_random(self.dist)


class Deterministic(PolicyDist):
    def __init__(self, dist):
        super(Deterministic, self).__init__(dist)
        ## For deterministic action, we only support continuous ones
        assert dist.dtype == convert_np_dtype_to_dtype_("float32") \
            or dist.dtype == convert_np_dtype_to_dtype_("float64")

    def __call__(self):
        return self.dist


def discrete_dist(input, mlp, exploration_rate=0.0):
    """
    Given an input and an MLP (list of layer funcs), this function
    returns a DiscreteDist object.
    The exploration_rate can be a number of a Variable.
    If it's a number, then the exploration rate is always fixed.
    """
    assert isinstance(exploration_rate, float) \
        or isinstance(exploration_rate, Variable)
    assert isinstance(mlp, list)
    dist = comf.feedforward(input, mlp)
    dist = comf.sum_to_one_norm_layer(dist + exploration_rate)
    return DiscreteDist(dist)


def q_discrete_dist(q_value, exploration_rate=0.0):
    """
    Generate a PolicyDist object given a Q value.
    We first construct a one-hot distribution according to the Q value,
    and then call discrete_dist().
    """
    assert len(q_value.shape) == 2, "[batch_size, num_actions]"
    max_id = comf.maxid_layer(q_value)
    prob = layers.cast(
        x=layers.one_hot(
            input=max_id, depth=q_value.shape[-1]),
        dtype="float32")
    return discrete_dist(prob, [], exploration_rate)
