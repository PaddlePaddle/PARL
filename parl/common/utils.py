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

from copy import deepcopy
import numpy as np


def split_list(l, sizes):
    """
    Split a list into several chunks, each chunk with a size in `sizes`.
    """
    chunks = []
    offset = 0
    for size in sizes:
        chunks.append(l[offset:offset + size])
        offset += size
    return chunks


def concat_dicts(dict_list):
    """
    Concatenate values of each key from a list of dictionary. 
    
    The type of values should be `numpy.ndarray`, and the result is
    the concatenation of these values at axis=0.
    """
    D = {}
    for d in dict_list:
        if not D:
            D = deepcopy(d)
        else:
            assert (d.viewkeys() == D.viewkeys())
            for k in D:
                assert isinstance(d[k], type(D[k]))
                if type(d[k]) == list:
                    D[k] += d[k]
                elif type(d[k] == np.ndarray):
                    D[k] = np.concatenate([D[k], d[k]])
                else:
                    raise TypeError("only numpy.ndarray or list is accepted")

    return D


def split_dict(D, starts):
    """
    Inverse operation of `concat_dicts`.
    """
    ret = []
    for i in range(len(starts) - 1):
        d = {}
        for k, v in D.iteritems():
            d[k] = deepcopy(v[starts[i]:starts[i + 1]])
        ret.append(d)
    return ret
