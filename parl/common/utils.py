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
from parl.common.error_handling import check_type_error


def concat_dicts(dicts):
    D = {}
    starts = [0]
    for d in dicts:
        if not D:
            D = deepcopy(d)
        else:
            assert (d.viewkeys() == D.viewkeys())
            for k in D:
                check_type_error(type(D[k]), type(d[k]))
                if type(d[k]) == list:
                    D[k] += d[k]
                elif type(d[k] == np.ndarray):
                    D[k] = np.concatenate([D[k], d[k]])
                else:
                    raise TypeError("only numpy.ndarray or list is accepted")

        sz = -1
        for v in D.itervalues():
            l = len(v) if type(v) == list else v.shape[0]
            if sz < 0:
                sz = l
            else:
                assert (sz == l)
        starts.append(sz)

    return D, starts


def split_dict(D, starts):
    ret = []
    for i in range(len(starts) - 1):
        d = {}
        for attr in D:
            d[attr] = D[attr][starts[i]:starts[i + 1]]
        ret.append(d)
    return ret
