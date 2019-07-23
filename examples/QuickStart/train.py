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
import parl
from cartpole_agent import CartpoleAgent
from cartpole_model import CartpoleModel
from parl.utils import logger
from utils import calc_discount_norm_reward

OBS_DIM = 4
ACT_DIM = 2
GAMMA = 0.99
LEARNING_RATE = 1e-3
SEED = 1


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


def main():
    env = gym.make("CartPole-v0")
    env.seed(SEED)
    np.random.seed(SEED)
    model = CartpoleModel(act_dim=ACT_DIM)
    alg = parl.algorithms.PolicyGradient(model, lr=LEARNING_RATE)
    agent = CartpoleAgent(alg, obs_dim=OBS_DIM, act_dim=ACT_DIM, seed=SEED)

    for i in range(1000):
        obs_list, action_list, reward_list = run_train_episode(env, agent)
        logger.info("Episode {}, Reward Sum {}.".format(i, sum(reward_list)))

        batch_obs = np.array(obs_list)
        batch_action = np.array(action_list)
        batch_reward = calc_discount_norm_reward(reward_list, GAMMA)

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
