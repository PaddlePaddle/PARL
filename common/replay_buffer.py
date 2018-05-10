import copy
import random
from collections import namedtuple

class Experience(object):
    def __init__(self, obs, action, reward, done):
        self.obs = obs
        self.reward = reward
        self.action = action
        self.done = done

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
        self._buffer = []           # a circular queue to store experiences
        self._capacity = capacity   # capacity of the buffer
        self._last = -1             # the index of the last element in the buffer
        self._exp_type = exp_type   # Experience class used in the buffer

    def __len__(self):
        return len(self._buffer)

    def has_next(self, i):
        return i != self._last

    def next_idx(self, i):
        if not self.has_next(i):
            return -1
        else:
            return (i+1) % self._capacity

    def add(self, exp):
        """
        Store one experience into the buffer.

        Args:
            exp(Experience): the experience to store in the buffer.
        """
        assert(isinstance(exp, self._exp_type))

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
                idx = random.randint(0, len(self._buffer)-1)
                if self.has_next(idx) and not self._buffer[idx].done:
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
            if not self.has_next(p) or self._buffer[p].done:
                print("warning")
            # make a copy of the buffer element as e may be modified somewhere
            e = copy.copy(self._buffer[p])
            exps.append(e)
            p = self.next_idx(p)

        return exps

