# Third party code
#
# The following code are copied or modified from:
# https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/optimizers.py

import numpy as np


class Optimizer(object):
    def __init__(self, parameter_size):
        self.dim = parameter_size
        self.t = 0

    def update(self, theta, global_g):
        self.t += 1
        step = self._compute_step(global_g)
        ratio = np.linalg.norm(step) / np.linalg.norm(theta)
        return theta + step, ratio

    def _compute_step(self, global_g):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, parameter_size, stepsize, momentum=0.9):
        Optimizer.__init__(self, parameter_size)
        self.v = np.zeros(self.dim, dtype=np.float32)
        self.stepsize, self.momentum = stepsize, momentum

    def _compute_step(self, global_g):
        self.v = self.momentum * self.v + (1. - self.momentum) * global_g
        step = -self.stepsize * self.v
        return step


class Adam(Optimizer):
    def __init__(self,
                 parameter_size,
                 stepsize,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-08):
        Optimizer.__init__(self, parameter_size)
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)

    def _compute_step(self, global_g):
        a = self.stepsize * (np.sqrt(1 - self.beta2**self.t) /
                             (1 - self.beta1**self.t))
        self.m = self.beta1 * self.m + (1 - self.beta1) * global_g
        self.v = self.beta2 * self.v + (1 - self.beta2) * (global_g * global_g)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step
