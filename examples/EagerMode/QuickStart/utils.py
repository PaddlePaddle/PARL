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


def calc_discount_norm_reward(reward_list, gamma):
    '''
    Calculate the discounted reward list according to the discount factor gamma, and normalize it.
    Args:
        reward_list(list): a list containing the rewards along the trajectory.
        gamma(float): the discounted factor for accumulation reward computation.
    Returns:
        a list containing the discounted reward
    '''
    discount_norm_reward = np.zeros_like(reward_list)

    discount_cumulative_reward = 0
    for i in reversed(range(0, len(reward_list))):
        discount_cumulative_reward = (
            gamma * discount_cumulative_reward + reward_list[i])
        discount_norm_reward[i] = discount_cumulative_reward
    discount_norm_reward = discount_norm_reward - np.mean(discount_norm_reward)
    discount_norm_reward = discount_norm_reward / np.std(discount_norm_reward)
    return discount_norm_reward
