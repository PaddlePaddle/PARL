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

import numpy as np
import scipy.signal

__all__ = ['calc_discount_sum_rewards', 'calc_gae', "Scaler"]


def calc_discount_sum_rewards(rewards, gamma):
    """ Calculate discounted forward sum of a sequence at each point. 

    Args:
        rewards (List/Tuple/np.array): rewards of (s_t, s_{t+1}, ..., s_T)
        gamma (Scalar): gamma coefficient

    Returns:
        np.array: discounted sum rewards of (s_t, s_{t+1}, ..., s_T)
    """
    return scipy.signal.lfilter([1.0], [1.0, -gamma], rewards[::-1])[::-1]


def calc_gae(rewards, values, next_value, gamma, lam):
    """ Calculate generalized advantage estimator (GAE).
    See: https://arxiv.org/pdf/1506.02438.pdf

    Args:
        rewards (List/Tuple/np.array): rewards of (s_t, s_{t+1}, ..., s_T)
        values (List/Tuple/np.array): values of (s_t, s_{t+1}, ..., s_T)
        next_value (Scalar): value of s_{T+1}
        gamma (Scalar): gamma coefficient
        lam (Scalar): lambda coefficient

    Returns:
        advantages (np.array): advantages of (s_t, s_{t+1}, ..., s_T)
    """
    # temporal differences
    tds = rewards + gamma * np.append(values[1:], next_value) - values
    advantages = calc_discount_sum_rewards(tds, gamma * lam)
    return advantages


class Scaler(object):
    """ Generate scale and offset based on running mean and stddev along axis=0

        offset = running mean
        scale = 1 / (stddev + 0.1) / 3 (i.e. 3x stddev = +/- 1.0)
    """

    def __init__(self, obs_dim):
        """
        Args:
            obs_dim: dimension of axis=1
        """
        self.vars = np.zeros(obs_dim)
        self.means = np.zeros(obs_dim)
        self.cnt = 0
        self.first_pass = True

    def update(self, x):
        """ Update running mean and variance (this is an exact method)
        Args:
            x: NumPy array, shape = (N, obs_dim)

        see: https://stats.stackexchange.com/questions/43159/how-to-calculate-pooled-
               variance-of-two-groups-given-known-group-variances-mean
        """
        if self.first_pass:
            self.means = np.mean(x, axis=0)
            self.vars = np.var(x, axis=0)
            self.cnt = x.shape[0]
            self.first_pass = False
        else:
            n = x.shape[0]
            new_data_var = np.var(x, axis=0)
            new_data_mean = np.mean(x, axis=0)
            new_data_mean_sq = np.square(new_data_mean)
            new_means = ((self.means * self.cnt) + (new_data_mean * n)) / (self.cnt + n)
            self.vars = (((self.cnt * (self.vars + np.square(self.means))) +
                          (n * (new_data_var + new_data_mean_sq))) / (self.cnt + n) - np.square(new_means))
            self.vars = np.maximum(0.0, self.vars)  # occasionally goes negative, clip
            self.means = new_means
            self.cnt += n

    def get(self):
        """ returns 2-tuple: (scale, offset) """
        return 1 / (np.sqrt(self.vars) + 0.1) / 3, self.means
