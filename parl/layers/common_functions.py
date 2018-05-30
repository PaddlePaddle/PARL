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


def discrete_random(prob):
    """
    Sample an id based on discrete distribution prob
    """
    cumsum = layers.cumsum(x=prob)
    r = layers.uniform_random_batch_size_like(
        input=prob, min=0., max=1., shape=[-1])
    index = layers.reduce_sum(layers.cast(cumsum < r, 'int'), dim=-1)
    index = layers.reshape(index, index.shape + (1, ))
    return index


def maxid_layer(input):
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
