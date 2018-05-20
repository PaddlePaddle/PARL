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
    def __init__(self, sensor_inputs, states, actions, game_status):
        check_type_error(list, type(sensor_inputs))
        self.sensor_inputs = sensor_inputs  # (observation, reward)
        self.states = states  # other states
        self.actions = actions  # actions taken
        self.game_status = game_status  # game status, e.g., max_steps or
        # episode end reached
        self.next_exp = None  # copy of the next Experience

    def set_next_exp(self, next_exp):
        self.next_exp = deepcopy(next_exp)

    #TODO: write copy function


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
    def __init__(self, capacity, exp_type=Experience):
        """
        Create Replay buffer.

        Args:
            exp_type(object): Experience class used in the buffer.
            capacity(int): Max number of experience to store in the buffer. When
                the buffer overflows the old memories are dropped.
        """
        check_gt("capacity", capacity, [], 1)
        self.buffer = []  # a circular queue to store experiences
        self.capacity = capacity  # capacity of the buffer
        self.last = -1  # the index of the last element in the buffer
        self.exp_type = exp_type  # Experience class used in the buffer

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
            exp(self.exp_type): the experience to store in the buffer.
        """
        check_type_error(self.exp_type, type(exp))
        # the next_exp field should be None at this point
        check_eq("next_exp", exp.next_exp, [], None)

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
                        idx].game_status:
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
                self.buffer_end(p) or self.buffer[p].game_status, p,
                self.buffer[p].game_status)
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

    @abstractmethod
    def add(self, exp):
        pass

    @abstractmethod
    def sample(self):
        pass


class ExperienceQueue(ExperienceQueueBase):
    def __init__(self, sample_seq):
        super(ExperienceQueue, self).__init__(sample_seq)
        self.q = deque()

    def __len__(self):
        return len(self.q)

    def add(self, exp):
        self.q.append(exp)

    def sample(self):
        exp_seqs = []
        while len(self.q) > 1:
            exps = []
            while not self.q[0].game_status and len(self.q) > 1:
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
            if self.q[0].game_status:
                self.q.popleft()
        return exp_seqs
