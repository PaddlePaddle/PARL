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

import argparse
import gym
import numpy as np
from cartpole_agent import CartpoleAgent
from cartpole_model import CartpoleModel
from parl.algorithms import PolicyGradient
from parl.utils import logger

OBS_DIM = 4
ACT_DIM = 2
GAMMA = 0.99
LEARNING_RATE = 1e-3


def run_train_episode(env, agent):
    obs_list, action_list, reward_list = [], [], []
    obs = env.reset()
    while True:
        obs_list.append(obs)
        action = agent.sample(obs)
        action_list.append(action)

        obs, reward, done, info = env.step(action)
        reward_list.append(reward)

        if done:
            break
    return obs_list, action_list, reward_list


def run_evaluate_episode(env, agent):
    obs = env.reset()
    all_reward = 0
    while True:
        if args.eval_vis:
            env.render()
        action = agent.predict(obs)
        obs, reward, done, info = env.step(action)
        all_reward += reward
        if done:
            break
    return all_reward


def calc_discount_norm_reward(reward_list):
    discount_norm_reward = np.zeros_like(reward_list)

    discount_cumulative_reward = 0
    for i in reversed(range(0, len(reward_list))):
        discount_cumulative_reward = (
            GAMMA * discount_cumulative_reward + reward_list[i])
        discount_norm_reward[i] = discount_cumulative_reward
    discount_norm_reward = discount_norm_reward - np.mean(discount_norm_reward)
    discount_norm_reward = discount_norm_reward / np.std(discount_norm_reward)
    return discount_norm_reward


def main():
    env = gym.make("CartPole-v0")
    model = CartpoleModel(act_dim=ACT_DIM)
    alg = PolicyGradient(model, hyperparas={'lr': LEARNING_RATE})
    agent = CartpoleAgent(alg, obs_dim=OBS_DIM, act_dim=ACT_DIM)

    for i in range(500):
        obs_list, action_list, reward_list = run_train_episode(env, agent)
        logger.info("Episode {}, Reward Sum {}.".format(i, sum(reward_list)))

        batch_obs = np.array(obs_list)
        batch_action = np.array(action_list)
        batch_reward = calc_discount_norm_reward(reward_list)

        agent.learn(batch_obs, batch_action, batch_reward)
        if (i + 1) % 100 == 0:
            all_reward = run_evaluate_episode(env, agent)
            logger.info('Test reward: {}'.format(all_reward))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--eval_vis',
        action='store_true',
        help='if set, will visualize the game when evaluating')
    args = parser.parse_args()

    main()
