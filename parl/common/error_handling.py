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

class LastElementError(Exception):
    """
    Raised when the last element or an episode-end element is sampled.

    Attributes:
        idx(int): the index of the element being picked
        is_episode_end(bool): whether the element is an episode end
    """

    def __init__(self, idx, is_episode_end):
        self._idx = idx
        self._episode_end = is_episode_end
        self.message = 'The element at ' + str(idx) + ' is '
        if is_episode_end:
            self.message += 'an episode end.'
        else:
            self.message += 'the last element.'
