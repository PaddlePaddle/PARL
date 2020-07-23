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


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.elements = [None for _ in range(capacity)]
        self.tree = [0 for _ in range(2 * capacity - 1)]
        self._ptr = 0

    def full(self):
        return all(self.elements)  # no `None` in self.elements

    def add(self, item, priority):
        self.elements[self._ptr] = item
        tree_idx = self._ptr + self.capacity - 1
        self.update(tree_idx, priority)
        self._ptr = (self._ptr + 1) % self.capacity

    def update(self, tree_idx, priority):
        diff = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) >> 1
            self.tree[tree_idx] += diff

    def retrieve(self, value):
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if value <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    value -= self.tree[left_child_idx]
                    parent_idx = right_child_idx
        elem_idx = leaf_idx - self.capacity + 1
        priority = self.tree[leaf_idx]
        return self.elements[elem_idx], leaf_idx, priority

    def from_list(self, lst):
        assert len(lst) == self.capacity
        self.elements = list(lst)
        for i in range(self.capacity - 1, 2 * self.capacity - 1):
            self.update(i, 1.0)

    @property
    def total_p(self):
        return self.tree[0]
