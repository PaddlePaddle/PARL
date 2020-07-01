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
import time

from .utils import SumTree, _check_full


class ProportionalPER:
    """Rank-based Prioritized Experience Replay.
    """

    def __init__(self, alpha, seg_num, size=1e6, eps=0.01, init_mem=None):
        self.size = int(size)
        self.eps = eps
        self.seg_num = seg_num
        self.alpha = alpha
        self.elements = SumTree(capacity=self.size)
        if init_mem:
            self.elements.from_list(init_mem)
        self._max_priority = 1.0

    def store(self, item, delta=None):
        assert len(item) == 5  # (s, a, r, s', terminal)
        if not delta:
            delta = self._max_priority
        assert delta > 0
        ps = np.power(delta + self.eps, self.alpha)
        self.elements.add(item, ps)

    def _clip_priorities(self, priorities):
        clipped = np.array([np.clip(p, -1, 1) for p in priorities])
        return clipped

    def update(self, indices, priorities):
        priorities = np.array(priorities) + self.eps
        priorities_alpha = np.power(priorities, self.alpha)
        for idx, priority in zip(indices, priorities_alpha):
            self.elements.update(idx, priority)
            self._max_priority = max(priority, self._max_priority)

    @_check_full
    def sample_one(self):
        sample_val = np.random.uniform(0, self.elements.total_p)
        item, tree_idx, _ = self.elements.retrieve(sample_val)
        return item, tree_idx

    @_check_full
    def sample(self):
        """ sample a batch of `seg_num` transitions

        Return:
            items: 
            indices: 
            probs: `N * P(i)`, for later calculating ISweights
        """
        seg_size = self.elements.total_p / self.seg_num
        seg_bound = [(seg_size * i, seg_size * (i + 1))
                     for i in range(self.seg_num)]
        items, indices, priorities = [], [], []
        for low, high in seg_bound:
            sample_val = np.random.uniform(low, high)
            item, tree_idx, priority = self.elements.retrieve(sample_val)
            items.append(item)
            indices.append(tree_idx)
            priorities.append(priority)

        probs = self.size * np.array(priorities) / self.elements.total_p
        return np.array(items), np.array(indices), np.array(probs)
