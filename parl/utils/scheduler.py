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

import six

__all__ = ['PiecewiseScheduler']


class PiecewiseScheduler(object):
    def __init__(self, scheduler_list):
        """ Piecewise scheduler of hyper parameter.

        Args:
            scheduler_list: list of (step, value) pair. E.g. [(0, 0.001), (10000, 0.0005)]
        """
        assert len(scheduler_list) > 0

        for i in six.moves.range(len(scheduler_list) - 1):
            assert scheduler_list[i][0] < scheduler_list[i + 1][0], \
                    'step of scheduler_list should be incremental.'

        self.scheduler_list = scheduler_list

        self.cur_index = 0
        self.cur_step = 0
        self.cur_value = self.scheduler_list[0][1]

        self.scheduler_num = len(self.scheduler_list)

    def step(self):
        """ Step one and fetch value according to following rule:

        Given scheduler_list: [(step_0, value_0), (step_1, value_1), ..., (step_N, value_N)],
        function will return value_K which satisfying self.cur_step >= step_K and self.cur_step < step_K+1
        """

        if self.cur_index < self.scheduler_num - 1:
            if self.cur_step >= self.scheduler_list[self.cur_index + 1][0]:
                self.cur_index += 1
                self.cur_value = self.scheduler_list[self.cur_index][1]

        self.cur_step += 1

        return self.cur_value
