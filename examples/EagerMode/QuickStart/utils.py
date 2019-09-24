import numpy as np


def calc_discount_norm_reward(reward_list, gamma):
    discount_norm_reward = np.zeros_like(reward_list)

    discount_cumulative_reward = 0
    for i in reversed(range(0, len(reward_list))):
        discount_cumulative_reward = (gamma * discount_cumulative_reward +
                                      reward_list[i])
        discount_norm_reward[i] = discount_cumulative_reward
    discount_norm_reward = discount_norm_reward - np.mean(discount_norm_reward)
    discount_norm_reward = discount_norm_reward / np.std(discount_norm_reward)
    return discount_norm_reward
