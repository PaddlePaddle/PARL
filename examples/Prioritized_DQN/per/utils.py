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

import functools
import queue

import numpy as np


class IndexPriorityQueue:
    """Indexed priority queue implemented with binary max heap.

    Index is introduced to help update priority of the sampled transition.
    `self.elements` is the unsorted list of elements, 
    `self.idx_heap` is the heap with element `[priority, idx]`, 
    in which `idx` is the index of the corresponding element in `self.elements`.
    """

    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.elements = [None for _ in range(self.max_size)]
        self.idx_heap = []  # [priority, idx] pair
        self._idx_pos = {}  # elements[idx]'s postion in `idx_heap`
        self._ptr = 0

    def size(self):
        return len(self.idx_heap)

    def full(self):
        return all(self.elements)

    def empty(self):
        return len(self.idx_heap) == 0

    def get_pos(self, idx):
        """Get the self.elements[idx]'s postion in `idx_heap`"""
        return self._idx_pos[idx]

    def put(self, item, priority):
        """Put item with priority
        Return: 
            idx: index of the item in `self.elements`
        """
        self.elements[self._ptr] = item
        idx = self._ptr

        if idx in self._idx_pos:
            pos = self._idx_pos[idx]
            self.idx_heap[pos] = [priority, idx]
            self._siftup(pos)
        else:
            self.idx_heap.append([priority, idx])
            pos = len(self.idx_heap) - 1
            self._idx_pos[idx] = pos
            self._siftdown(0, pos)

        self._ptr = (idx + 1) % self.max_size
        return idx

    def update_item(self, idx, priority):
        pos = self.get_pos(idx)
        self.idx_heap[pos][0] = priority
        self._siftup(pos)

    def _siftdown(self, startpos, pos):
        item = self.idx_heap[pos]
        while pos > startpos:
            parentpos = (pos - 1) >> 1
            parent = self.idx_heap[parentpos]
            if item[0] > parent[0]:
                self.idx_heap[pos] = parent
                self._idx_pos[parent[1]] = pos
                pos = parentpos
                continue
            break
        self.idx_heap[pos] = item
        self._idx_pos[item[1]] = pos

    def _siftup(self, pos):
        endpos = len(self.idx_heap)
        startpos = pos
        item = self.idx_heap[pos]

        childpos = 2 * pos + 1
        while childpos < endpos:
            rightpos = childpos + 1
            if rightpos < endpos and \
                not self.idx_heap[childpos] > self.idx_heap[rightpos]:
                childpos = rightpos
            self.idx_heap[pos] = self.idx_heap[childpos]
            self._idx_pos[self.idx_heap[childpos][1]] = pos
            pos = childpos
            childpos = 2 * pos + 1
        self.idx_heap[pos] = item
        self._idx_pos[item[1]] = pos
        self._siftdown(startpos, pos)

    def from_list(self, lst):
        assert len(lst) == self.max_size
        self.elements = lst.copy()
        for idx, elem in enumerate(self.elements):
            self.idx_heap.append([1.0, idx])
            self._idx_pos[idx] = idx


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
        self.elements = lst.copy()
        for i in range(self.capacity - 1, 2 * self.capacity - 1):
            self.update(i, 1.0)

    @property
    def total_p(self):
        return self.tree[0]


def _check_full(func):
    @functools.wraps(func)
    def check_full(self, *args, **kwargs):
        try:
            assert self.elements.full()
        except AssertionError:
            raise RuntimeError("The replay memory is not full!")
        return func(self, *args, **kwargs)

    return check_full
