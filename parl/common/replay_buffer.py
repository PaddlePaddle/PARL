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
    """
    A Sample represents one or a sequence of Experiences
    """
    def __init__(self, i, n):
        self.i = i # starting index of the first experience in the sample
        self.n = n # length of the sequence

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
        assert(capacity > 1)
        self.buffer = []  # a circular queue to store experiences
        self.capacity = capacity  # capacity of the buffer
        self.last = -1  # the index of the last element in the buffer
        self.exp_type = exp_type  # Experience class used in the buffer

    def __len__(self):
        return len(self.buffer)

    def end_of_buffer(self, i):
        return i == self.last

    def next_idx(self, i):
        if self.end_of_buffer(i):
            return -1
        else:
            return (i + 1) % self.capacity

    def add(self, exp):
        """
        Store one experience into the buffer.

        Args:
            exp(self.exp_type): the experience to store in the buffer.
        """
        assert (isinstance(exp, self.exp_type))

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.last = (self.last + 1) % self.capacity
        self.buffer[self.last] = exp

    def sample(self, num_entries):
        """
        Generate a batch of Samples.

        Args:
            num_entries(int): Number of samples to generate.
            
        Returns(Sample): one sample of experiences.
        """

        for _ in xrange(num_entries):
            while True:
                idx = random.randint(0, len(self.buffer) - 1)
                if not self.end_of_buffer(idx) and not self.buffer[idx].episode_end:
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
            if self.end_of_buffer(p) or self.buffer[p].episode_end:
                raise LastElementError(p, self.buffer[p].episode_end)
            # make a copy of the buffer element as e may be modified somewhere
            e = copy.copy(self.buffer[p])
            exps.append(e)
            p = self.next_idx(p)

        return exps
