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

__all__ = ['calc_discount_sum_rewards', 'calc_gae']


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
