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

from .utils import SumTree, BasePER


class ProportionalPER(BasePER):
    """Rank-based Prioritized Experience Replay.
    """

    def __init__(self,
                 alpha,
                 seg_num,
                 size=1e6,
                 eps=0.01,
                 init_mem=None,
                 framestack=4):
        super(ProportionalPER, self).__init__(
            alpha=alpha,
            seg_num=seg_num,
            elem_cls=SumTree,
            size=size,
            init_mem=init_mem,
            framestack=framestack)
        self.eps = eps

    def store(self, item, delta=None):
        assert len(item) == 5  # (s, a, r, s', terminal)
        if not delta:
            delta = self._max_priority
        assert delta >= 0
        ps = np.power(delta + self.eps, self.alpha)
        self.elements.add(item, ps)

    def update(self, indices, priorities):
        priorities = np.array(priorities) + self.eps
        priorities_alpha = np.power(priorities, self.alpha)
        for idx, priority in zip(indices, priorities_alpha):
            self.elements.update(idx, priority)
            self._max_priority = max(priority, self._max_priority)

    def sample_one(self):
        assert self.elements.full(), "The replay memory is not full!"
        sample_val = np.random.uniform(0, self.elements.total_p)
        item, tree_idx, _ = self.elements.retrieve(sample_val)
        return item, tree_idx

    def sample(self):
        """ sample a batch of `seg_num` transitions

        Return:
            items: 
            indices: 
            probs: `N * P(i)`, for later calculating sampling weights
        """
        assert self.elements.full(), "The replay memory is not full!"
        seg_size = self.elements.total_p / self.seg_num
        seg_bound = [(seg_size * i, seg_size * (i + 1))
                     for i in range(self.seg_num)]
        items, indices, priorities = [], [], []
        for low, high in seg_bound:
            sample_val = np.random.uniform(low, high)
            _, tree_idx, priority = self.elements.retrieve(sample_val)
            elem_idx = tree_idx - self.elements.capacity + 1
            item = self._get_stacked_item(elem_idx)
            items.append(item)
            indices.append(tree_idx)
            priorities.append(priority)

        probs = self.size * np.array(priorities) / self.elements.total_p
        return np.array(items), np.array(indices), np.array(probs)
