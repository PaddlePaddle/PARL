#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#-*- coding: utf-8 -*-

# 检查版本
import gym
import parl
import paddle
assert paddle.__version__ == "2.2.0", "[Version WARNING] please try `pip install paddlepaddle==2.2.0`"
assert parl.__version__ == "2.0.3", "[Version WARNING] please try `pip install parl==2.0.1`"
assert gym.__version__ == "0.18.0", "[Version WARNING] please try `pip install gym==0.18.0`"

import os
import gym
import numpy as np
import parl

from agent import Agent
from model import Model
from algorithm import PolicyGradient  # from parl.algorithms import PolicyGradient

from parl.utils import logger

LEARNING_RATE = 1e-3


# 训练一个episode
def run_train_episode(agent, env):
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


# 评估 agent, 跑 5 个episode，总reward求平均
def run_evaluate_episodes(agent, env, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        while True:
            action = agent.predict(obs)
            obs, reward, isOver, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if isOver:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


def calc_reward_to_go(reward_list, gamma=1.0):
    for i in range(len(reward_list) - 2, -1, -1):
        # G_i = r_i + γ·G_i+1
        reward_list[i] += gamma * reward_list[i + 1]  # Gt
    return np.array(reward_list)


def main():
    env = gym.make('CartPole-v0')
    # env = env.unwrapped # Cancel the minimum score limit
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

    # 根据parl框架构建agent
    model = Model(obs_dim=obs_dim, act_dim=act_dim)
    alg = PolicyGradient(model, lr=LEARNING_RATE)
    agent = Agent(alg)

    # 加载模型并评估
    # if os.path.exists('./model.ckpt'):
    #     agent.restore('./model.ckpt')
    #     run_evaluate_episodes(agent, env, render=True)
    #     exit()

    for i in range(1000):
        obs_list, action_list, reward_list = run_train_episode(agent, env)
        if i % 10 == 0:
            logger.info("Episode {}, Reward Sum {}.".format(
                i, sum(reward_list)))

        batch_obs = np.array(obs_list)
        batch_action = np.array(action_list)
        batch_reward = calc_reward_to_go(reward_list)

        agent.learn(batch_obs, batch_action, batch_reward)
        if (i + 1) % 100 == 0:
            # render=True 查看显示效果
            total_reward = run_evaluate_episodes(agent, env, render=False)
            logger.info('Test reward: {}'.format(total_reward))

    # save the parameters to ./model.ckpt
    agent.save('./model.ckpt')


if __name__ == '__main__':
    main()
