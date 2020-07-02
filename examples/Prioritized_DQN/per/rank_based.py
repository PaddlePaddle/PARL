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

import time
import functools
import random

import numpy as np
from .utils import IndexPriorityQueue, BasePER


class RankPER(BasePER):
    """Rank-based Prioritized Experience Replay.
    """

    def __init__(self, alpha, seg_num, size=1e6, init_mem=None, framestack=4):
        super(RankPER, self).__init__(
            alpha=alpha,
            seg_num=seg_num,
            elem_cls=IndexPriorityQueue,
            size=size,
            init_mem=init_mem,
            framestack=framestack)
        self.pdf = self._build_pdf(heap_size=self.size)
        self.seg_bound = self._compute_seg_bound()

    def _build_pdf(self, heap_size):
        p = np.array([1 / (rank + 1) for rank in range(heap_size)])
        p_alpha = np.power(p, self.alpha)
        pdf = p_alpha / np.sum(p_alpha)
        return pdf

    def _compute_seg_bound(self):
        """Precompute segment boundaries.

        Args:

        Return:
            seg_bound: {seg: end_idx}, the range of one segment is 
                        [seg_bound[seg-1]: seg_bound[seg]]
        """
        cdf = np.cumsum(self.pdf)
        seg_bound = {i: None for i in range(1, self.seg_num + 1)}
        cur_bound = seg_size = 1 / self.seg_num
        index = 0
        for seg in list(seg_bound.keys()):
            while index < len(cdf) and cdf[index] < cur_bound:
                index += 1
            seg_bound[seg] = index + 1
            cur_bound += seg_size
        seg_bound[0] = 0
        return seg_bound

    def _get_seg(self, seg_id):
        assert seg_id > 0
        start, end = self.seg_bound[seg_id - 1], self.seg_bound[seg_id]
        if start == end:
            start -= 1
        segment = self.elements.idx_heap[start:end]
        return segment

    def _sample_from_segment(self, seg_id):
        segement = self._get_seg(seg_id)
        _, sampled_idx = random.choice(segement)
        item = self._get_stacked_item(sampled_idx)
        prob = self.pdf[sampled_idx]
        return item, sampled_idx, prob

    def store(self, item, delta=None):
        assert len(item) == 5  # [s, a, r, s', terminal]
        if not delta:
            delta = self._max_priority
        assert delta >= 0
        idx = self.elements.put(item, delta)
        return idx

    def update(self, idxs, priorities):
        assert len(idxs) == len(priorities)
        for idx, priority in zip(idxs, priorities):
            self.elements.update_item(idx, priority)
            self._max_priority = max(priority, self._max_priority)

    def sample_one(self):
        """Sample one item from heap.
         
        First sample one segment, then sample from the segment uniformly.
        """
        self._check_full()
        seg_id = np.random.choice([i for i in range(1, self.seg_num + 1)])
        item, idx, _ = self._sample_from_segment(seg_id)
        return item, idx

    def sample(self):
        """Sample `batch_size` items from heap.

        Return:
            items: batch of `k` transition, each from one segment
            idxs: indexes of sampled transition in `self.elements`
            probs: `N * P(i)`, for later calculating sampling weights
        """
        self._check_full()
        items, idxs, probs = [], [], []
        for seg_id in range(1, self.seg_num + 1):
            item, idx, prob = self._sample_from_segment(seg_id)
            items.append(item)
            idxs.append(idx)
            probs.append(prob)
        probs = self.size * np.array(probs)
        return np.array(items), np.array(idxs), np.array(probs)
