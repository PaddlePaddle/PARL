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


class Feedforward(layers.Network):
    """
     A feedforward network can contain a sequence of components,
     where each component can be either a LayerFunc or a Feedforward.
     The purpose of this class is to create a collection of LayerFuncs that can
     be easily copied from one Network to another.
     Examples of feedforward networks can be MLP and CNN.
     """

    def __init__(self, components):
        for i in range(len(components)):
            setattr(self, "ff%06d" % i, components[i])

    def __call__(self, input):
        attrs = {
            attr: getattr(self, attr)
            for attr in dir(self) if "ff" in attr
        }
        for k in sorted(attrs.keys()):
            input = attrs[k](input)
        return input


class MLP(Feedforward):
    def __init__(self, multi_fc_layers):
        super(MLP, self).__init__([layers.fc(**c) for c in multi_fc_layers])


class CNN(Feedforward):
    """
    Image CNN
    """

    def __init__(self, multi_conv_layers):
        super(CNN, self).__init__(
            [layers.conv2d(**c) for c in multi_conv_layers])


def category_random(prob):
    """
    Sample an id based on category distribution prob
    """
    cumsum = layers.cumsum(x=prob)
    r = layers.uniform_random_batch_size_like(
        input=prob, min=0., max=1., shape=[-1])
    index = layers.reduce_sum(layers.cast(cumsum < r, 'int'), dim=-1)
    index = layers.reshape(index, index.shape + (1, ))
    return index


def argmax_layer(input):
    """
    Get the id of the max val of an input vector
    """
    _, index = layers.topk(input, 1)
    return index


def inner_prod(x, y):
    """
    Get the inner product of two vectors
    """
    return layers.reduce_sum(layers.elementwise_mul(x, y), dim=-1)


def sum_to_one_norm_layer(input):
    eps = 1e-9  # avoid dividing 0
    sum = layers.reduce_sum(input + eps, dim=-1)
    return layers.elementwise_div(x=input, y=sum, axis=0)


def idx_select(input, idx):
    """
    Given an input vector (Variable) and an idx (int or Variable),
    select the entry of the vector according to the idx.
    """
    assert isinstance(input, Variable)
    assert len(input.shape) == 2
    batch_size, num_entries = input.shape

    if isinstance(idx, int):
        ## if idx is a constant int, then we create a variable
        idx = layers.fill_constant(
            shape=[batch_size, 1], dtype="int64", value=idx)
    else:
        assert isinstance(idx, Variable)

    assert input.shape
    select = layers.cast(
        x=layers.one_hot(
            input=idx, depth=num_entries), dtype="float32")
    return inner_prod(select, input)
