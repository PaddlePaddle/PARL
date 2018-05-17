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

import copy
import random
from error_handling import LastElementError


class Experience(object):
    def __init__(self, obs, action, reward, episode_end):
        self.obs = obs
        self.reward = reward
        self.action = action
        self.episode_end = episode_end


class Sample(object):
    def __init__(self, i, n):
        self.i = i
        self.n = n

    def __repr__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


class ReplayBuffer(object):
    def __init__(self, capacity, exp_type=Experience):
        """
        Create Replay buffer.

        Args:
            exp_type(object): Experience class used in the buffer.
            capacity(int): Max number of experience to store in the buffer. When
                the buffer overflows the old memories are dropped.
        """
        # TODO: check exp_type is a type of Experience
        self._buffer = []  # a circular queue to store experiences
        self._capacity = capacity  # capacity of the buffer
        self._last = -1  # the index of the last element in the buffer
        self._exp_type = exp_type  # Experience class used in the buffer

    def __len__(self):
        return len(self._buffer)

    def has_next(self, i):
        return i != self._last

    def next_idx(self, i):
        if not self.has_next(i):
            return -1
        else:
            return (i + 1) % self._capacity

    def add(self, exp):
        """
        Store one experience into the buffer.

        Args:
            exp(Experience): the experience to store in the buffer.
        """
        assert (isinstance(exp, self._exp_type))

        if len(self._buffer) < self._capacity:
            self._buffer.append(None)
        self._last = (self._last + 1) % self._capacity
        self._buffer[self._last] = exp

    def sample(self, num_entries):
        """
        Generate a batch of Samples.

        Args:
            num_entries(int): Number of samples to generate.
            
        Returns(Sample): one sample of experiences.
        """

        for _ in xrange(num_entries):
            while True:
                idx = random.randint(0, len(self._buffer) - 1)
                if self.has_next(idx) and not self._buffer[idx].episode_end:
                    break
            yield Sample(idx, 1)

    def get_experiences(self, sample):
        """
        Get experiences from a sample

        Args:
            sample(Sample): a sample of experiences

        Return(list): a list of Experiences
        """
        exps = []
        p = sample.i
        for _ in xrange(sample.n):
            if not self.has_next(p) or self._buffer[p].episode_end:
                raise LastElementError(p, self._buffer[p].episode_end)
            # make a copy of the buffer element as e may be modified somewhere
            e = copy.copy(self._buffer[p])
            exps.append(e)
            p = self.next_idx(p)

        return exps
