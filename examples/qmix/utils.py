#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np


class OneHotTransform(object):
    def __init__(self, out_dim):
        self.out_dim = out_dim

    def __call__(self, agent_id):
        assert agent_id < self.out_dim
        one_hot_id = np.zeros(self.out_dim, dtype='float32')
        one_hot_id[agent_id] = 1.0
        return one_hot_id


class AvailableActionsSampler(object):
    ''' Sample available actions uniformly.
    '''

    def __init__(self, data):
        '''data: np.ndarray (batch_size, len_distributions)'''
        assert len(data.shape) == 2
        self.batch_list = []
        for i in range(data.shape[0]):
            elements = set()
            for j in range(data.shape[1]):
                if np.abs(data[i, j] - 1.0) < 1e-5:
                    # add action idx
                    elements.add(j)
            self.batch_list.append(list(elements))

    def sample(self):
        results = []
        for i in range(len(self.batch_list)):
            candidates = self.batch_list[i]
            results.append(np.random.choice(candidates))
        return np.array(results, dtype='long')
