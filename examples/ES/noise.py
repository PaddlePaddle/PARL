# Third party code
#
# The following code are copied or modified from:
# https://github.com/ray-project/ray/blob/master/python/ray/rllib/utils/filter.py

import numpy as np


class SharedNoiseTable(object):
    """Shared noise table used by learner and actor.

    Learner and actor will create a same noise table by passing the same seed.
    With the same noise table, learner and actor can communicate the noises by
    index of noise table instead of numpy array of noises.
    """

    def __init__(self, noise_size, seed=1024):
        self.noise_size = noise_size
        self.seed = seed
        self.noise = self._create_noise()

    def _create_noise(self):
        noise = np.random.RandomState(self.seed).randn(self.noise_size).astype(
            np.float32)
        return noise

    def get(self, i, dim):
        return self.noise[i:i + dim]

    def sample_index(self, dim):
        return np.random.randint(0, len(self.noise) - dim + 1)
