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

from abc import ABCMeta, abstractmethod
from copy import deepcopy
from Queue import deque
import random
from parl.common.error_handling import *


class Experience(object):
    def set_next_exp(self, next_exp):
        self.next_exp = deepcopy(next_exp)

    @staticmethod
    def define(name, attrs):
        """
        Create an Experience 
        """

        def set_attributes(self, **kwargs):
            for k, v in kwargs.iteritems():
                if not hasattr(self, k):
                    raise TypeError
                setattr(self, k, v)

        check_type_error(list, type(attrs))
        cls_attrs = dict((attr, None) for attr in attrs)
        cls_attrs['next_exp'] = None  # add attribute "next_exp"
        # __init__ of the new Experience class
        cls_attrs['__init__'] = set_attributes
        cls = type(name, (Experience, ), cls_attrs)
        return cls


class Sample(object):
    """
    A Sample represents one or a sequence of Experiences
    """

    def __init__(self, i, n):
        self.i = i  # starting index of the first experience in the sample
        self.n = n  # length of the sequence

    def __repr__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


class ReplayBuffer(object):
    def __init__(self, capacity):
        """
        Create Replay buffer.

        Args:
            exp_type(object): Experience class used in the buffer.
            capacity(int): Max number of experience to store in the buffer. When
                the buffer overflows the old memories are dropped.
        """
        assert capacity > 1
        self.buffer = []  # a circular queue to store experiences
        self.capacity = capacity  # capacity of the buffer
        self.last = -1  # the index of the last element in the buffer

    def __len__(self):
        return len(self.buffer)

    def buffer_end(self, i):
        return i == self.last

    def next_idx(self, i):
        if self.buffer_end(i):
            return -1
        else:
            return (i + 1) % self.capacity

    def add(self, exp):
        """
        Store one experience into the buffer.

        Args:
            exp(Experience): the experience to store in the buffer.
        """
        # the next_exp field should be None at this point
        assert isinstance(exp, Experience)
        assert exp.next_exp is None

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.last = (self.last + 1) % self.capacity
        self.buffer[self.last] = deepcopy(exp)

    def sample(self, num_samples):
        """
        Generate a batch of Samples. Each Sample represents a sequence of
        Experiences (length>=1). And a sequence must not cross the boundary
        between two games. 

        Args:
            num_samples(int): Number of samples to generate.
            
        Returns: A generator of Samples
        """
        if len(self.buffer) <= 1:
            yield []

        for _ in xrange(num_samples):
            while True:
                idx = random.randint(0, len(self.buffer) - 1)
                if not self.buffer_end(idx) and not self.buffer[
                        idx].episode_end:
                    break
            yield Sample(idx, 1)

    def get_experiences(self, sample):
        """
        Get Experiences from a Sample

        Args:
            sample(Sample): a Sample representing a sequence of Experiences

        Return(list): a list of Experiences
        """
        exps = []
        p = sample.i
        for _ in xrange(sample.n):
            check_last_exp_error(
                self.buffer_end(p) or self.buffer[p].episode_end, p,
                self.buffer[p].episode_end)
            # make a copy of the buffer element as e may be modified somewhere
            e = deepcopy(self.buffer[p])
            p = self.next_idx(p)
            e.set_next_exp(self.buffer[p])
            exps.append(e)

        return exps


class ExperienceQueueBase(object):
    __metaclass__ = ABCMeta

    def __init__(self, sample_seq):
        self.sample_seq = sample_seq
        self.exp_cls = None

    def __pre_add(self, **kwargs):
        if not self.exp_cls:
            self.exp_cls = Experience.define("Experience", kwargs.keys())
        return self.exp_cls(**kwargs)

    def add(self, **kwargs):
        exp = self.__pre_add(**kwargs)
        self._add(exp)

    @abstractmethod
    def _add(self, exp):
        pass

    @abstractmethod
    def sample(self):
        pass


class NoReplacementQueue(ExperienceQueueBase):
    def __init__(self, sample_seq):
        super(NoReplacementQueue, self).__init__(sample_seq)
        self.q = deque()

    def __len__(self):
        return len(self.q)

    def _add(self, exp):
        self.q.append(exp)

    def sample(self):
        exp_seqs = []
        while len(self.q) > 1:
            exps = []
            while not self.q[0].episode_end and len(self.q) > 1:
                exps.append(self.q.popleft())
            if len(exps) > 0:
                for i in xrange(len(exps) - 1):
                    exps[i].next_exp = deepcopy(exps[i + 1])
                exps[-1].next_exp = deepcopy(self.q[0])
                if self.sample_seq:
                    exp_seqs.append(exps)
                else:
                    for e in exps:
                        exp_seqs.append([e])
            if self.q[0].episode_end:
                self.q.popleft()
        return exp_seqs
