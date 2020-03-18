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


def discretize(value, n_dim, min_val, max_val):
    '''
    discretize a value into a vector of n_dim dimension 1-hot representation
    with the value below min_val being [1, 0, 0, ..., 0]
    and the value above max_val being [0, 0, ..., 0, 1]
    '''
    assert n_dim > 0
    if (n_dim == 1):
        return [1]
    delta = (max_val - min_val) / float(n_dim - 1)
    active_pos = int((value - min_val) / delta + 0.5)
    active_pos = min(n_dim - 1, active_pos)
    active_pos = max(0, active_pos)
    ret_array = [0 for i in range(n_dim)]
    ret_array[active_pos] = 1.0
    return ret_array


def linear_discretize(value, n_dim, min_val, max_val):
    '''
    discretize a value into a vector of n_dim dimensional representation
    with the value below min_val being [1, 0, 0, ..., 0]
    and the value above max_val being [0, 0, ..., 0, 1]
    e.g. if n_dim = 2, min_val = 1.0, max_val = 2.0
      if value  = 1.5 returns [0.5, 0.5], if value = 1.8 returns [0.2, 0.8]
    '''
    assert n_dim > 0
    if (n_dim == 1):
        return [1]
    delta = (max_val - min_val) / float(n_dim - 1)
    active_pos = int((value - min_val) / delta + 0.5)
    active_pos = min(n_dim - 2, active_pos)
    active_pos = max(0, active_pos)
    anchor_pt = active_pos * delta + min_val
    if (anchor_pt > value and anchor_pt > min_val + 0.5 * delta):
        anchor_pt -= delta
        active_pos -= 1
    weight = (value - anchor_pt) / delta
    weight = min(1.0, max(0.0, weight))
    ret_array = [0 for i in range(n_dim)]
    ret_array[active_pos] = 1.0 - weight
    ret_array[active_pos + 1] = weight
    return ret_array
