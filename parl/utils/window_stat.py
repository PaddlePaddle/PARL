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

__all__ = ['WindowStat']

import numpy as np


class WindowStat(object):
    """ Tool to maintain statistical data in a window.
    """

    def __init__(self, window_size):
        self.items = [None] * window_size
        self.idx = 0
        self.count = 0

    def add(self, obj):
        self.items[self.idx] = obj
        self.idx += 1
        self.count += 1
        self.idx %= len(self.items)

    @property
    def mean(self):
        if self.count > 0:
            return np.mean(self.items[:self.count])
        else:
            return None

    @property
    def min(self):
        if self.count > 0:
            return np.min(self.items[:self.count])
        else:
            return None

    @property
    def max(self):
        if self.count > 0:
            return np.max(self.items[:self.count])
        else:
            return None
