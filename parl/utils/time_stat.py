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

import time
from parl.utils.window_stat import WindowStat

__all_ = ['TimeStat']


class TimeStat(object):
    """A time stat for logging the elapsed time of code running

    Example:
        time_stat = TimeStat()
        with time_stat:
            // some code
        print(time_stat.mean)
    """

    def __init__(self, window_size=1):
        self.time_samples = WindowStat(window_size)
        self._start_time = None

    def __enter__(self):
        self._start_time = time.time()

    def __exit__(self, type, value, tb):
        time_delta = time.time() - self._start_time
        self.time_samples.add(time_delta)

    @property
    def mean(self):
        return self.time_samples.mean

    @property
    def min(self):
        return self.time_samples.min

    @property
    def max(self):
        return self.time_samples.max
