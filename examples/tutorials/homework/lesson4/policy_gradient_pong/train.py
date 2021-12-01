#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

# 检查paddle和parl的版本
import gym
import parl
import paddle
assert paddle.__version__ == "1.6.3", "[Version WARNING] please try `pip install paddlepaddle==1.6.3`"
assert parl.__version__ == "1.3.1" or parl.__version__ == "1.4", "[Version WARNING] please try `pip install parl==1.3.1` or `pip install parl==1.4` "
assert gym.__version__ == "0.18.0", "[Version WARNING] please try `pip install gym==0.18.0`"

import os
import gym
import numpy as np
import parl

from agent import Agent
from model import Model
from parl.algorithms import PolicyGradient

from parl.utils import logger

LEARNING_RATE = 1e-3


def run_episode(env, agent):
    obs_list, action_list, reward_list = [], [], []
    obs = env.reset()
    while True:
        obs = preprocess(obs)  # from shape (210, 160, 3) to (100800,)
        obs_list.append(obs)
        action = agent.sample(obs)
        action_list.append(action)

        obs, reward, done, info = env.step(action)
        reward_list.append(reward)

        if done:
            break
    return obs_list, action_list, reward_list


# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        while True:
            obs = preprocess(obs)  # from shape (210, 160, 3) to (100800,)
            action = agent.predict(obs)
            obs, reward, isOver, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if isOver:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


def preprocess(image):
    """ 预处理 210x160x3 uint8 frame into 6400 (80x80) 1维 float vector """
    image = image[35:195]  # 裁剪
    image = image[::2, ::2, 0]  # 下采样，缩放2倍
    image[image == 144] = 0  # 擦除背景 (background type 1)
    image[image == 109] = 0  # 擦除背景 (background type 2)
    image[image != 0] = 1  # 转为灰度图，除了黑色外其他都是白色
    return image.astype(np.float).ravel()


def calc_reward_to_go(reward_list, gamma=0.99):
    """calculate discounted reward"""
    reward_arr = np.array(reward_list)
    for i in range(len(reward_arr) - 2, -1, -1):
        # G_t = r_t + γ·r_t+1 + ... = r_t + γ·G_t+1
        reward_arr[i] += gamma * reward_arr[i + 1]
    # normalize episode rewards
    reward_arr -= np.mean(reward_arr)
    reward_arr /= np.std(reward_arr)
    return reward_arr


def main():
    env = gym.make('Pong-v0')
    obs_dim = 80 * 80
    act_dim = env.action_space.n
    logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

    # 根据parl框架构建agent
    model = Model(act_dim=act_dim)
    alg = PolicyGradient(model, lr=LEARNING_RATE)
    agent = Agent(alg, obs_dim=obs_dim, act_dim=act_dim)

    # 加载模型
    # if os.path.exists('./model.ckpt'):
    #     agent.restore('./model.ckpt')

    for i in range(1000):
        obs_list, action_list, reward_list = run_episode(env, agent)
        if i % 10 == 0:
            logger.info("Train Episode {}, Reward Sum {}.".format(
                i, sum(reward_list)))

        batch_obs = np.array(obs_list)
        batch_action = np.array(action_list)
        batch_reward = calc_reward_to_go(reward_list)

        agent.learn(batch_obs, batch_action, batch_reward)
        if (i + 1) % 100 == 0:
            total_reward = evaluate(env, agent, render=False)
            logger.info('Episode {}, Test reward: {}'.format(
                i + 1, total_reward))

    # save the parameters to ./model.ckpt
    agent.save('./model.ckpt')


if __name__ == '__main__':
    main()
