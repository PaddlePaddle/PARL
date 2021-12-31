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

# -*- coding: utf-8 -*-

# 检查paddle和parl的版本
import gym
import parl
import paddle
assert paddle.__version__ == "1.6.3", "[Version WARNING] please try `pip install paddlepaddle==1.6.3`"
assert parl.__version__ == "1.3.1" or parl.__version__ == "1.4", "[Version WARNING] please try `pip install parl==1.3.1` or `pip install parl==1.4` "
assert gym.__version__ == "0.18.0", "[Version WARNING] please try `pip install gym==0.18.0`"

import os
import numpy as np

import parl
from parl import layers
from paddle import fluid
from parl.utils import logger
from parl.utils import ReplayMemory  # 经验回放

from rlschool import make_env  # 使用 RLSchool 创建飞行器环境
from quadrotor_model import QuadrotorModel
from quadrotor_agent import QuadrotorAgent
from parl.algorithms import DDPG

GAMMA = 0.99  # reward 的衰减因子，一般取 0.9 到 0.999 不等
TAU = 0.001  # target_model 跟 model 同步参数 的 软更新参数
ACTOR_LR = 0.0002  # Actor网络更新的 learning rate
CRITIC_LR = 0.001  # Critic网络更新的 learning rate
MEMORY_SIZE = 1e6  # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 1e4  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
REWARD_SCALE = 0.01  # reward 的缩放因子
BATCH_SIZE = 256  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
TRAIN_TOTAL_STEPS = 1e6  # 总训练步数
TEST_EVERY_STEPS = 1e4  # 每个N步评估一下算法效果，每次评估5个episode求平均reward


def action_mapping(model_output_act, low_bound, high_bound):
    """ 将模型输出的动作从范围 [-1,1] 映射到 [low_bound, high_bound].
    参数:
        model_output_act: np.array, 值的范围为[-1, 1]
        low_bound: float, 环境动作空间的下界
        high_bound: float, 环境动作空间的上界
    返回:
        action: np.array, 值的范围为 [low_bound, high_bound]
    """
    assert np.all(((model_output_act<=1.0), (model_output_act>=-1.0))), \
        'the action should be in range [-1.0, 1.0]'
    assert high_bound > low_bound
    action = low_bound + (model_output_act - (-1.0)) * (
        (high_bound - low_bound) / 2.0)
    action = np.clip(action, low_bound, high_bound)
    return action


def run_episode(env, agent, rpm):
    obs = env.reset()
    total_reward, steps = 0, 0
    while True:
        steps += 1
        batch_obs = np.expand_dims(obs, axis=0)
        action = agent.predict(batch_obs.astype('float32'))
        action = np.squeeze(action)

        # Add exploration noise, and clip to [-1.0, 1.0]
        action = np.clip(np.random.normal(action, 1.0), -1.0, 1.0)

        real_action = action_mapping(action, env.action_space.low[0],
                                     env.action_space.high[0])
        next_obs, reward, done, info = env.step(real_action)
        rpm.append(obs, action, REWARD_SCALE * reward, next_obs, done)

        if rpm.size() > MEMORY_WARMUP_SIZE:
            batch_obs, batch_action, batch_reward, batch_next_obs, \
                    batch_terminal = rpm.sample_batch(BATCH_SIZE)
            critic_cost = agent.learn(batch_obs, batch_action, batch_reward,
                                      batch_next_obs, batch_terminal)

        obs = next_obs
        total_reward += reward

        if done:
            break
    return total_reward, steps


# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        total_reward, steps = 0, 0
        while True:
            batch_obs = np.expand_dims(obs, axis=0)
            action = agent.predict(batch_obs.astype('float32'))
            action = np.squeeze(action)
            action = np.clip(action, -1.0, 1.0)  ## special

            real_action = action_mapping(action, env.action_space.low[0],
                                         env.action_space.high[0])
            next_obs, reward, done, info = env.step(real_action)

            obs = next_obs
            total_reward += reward
            steps += 1

            if render:
                env.render()

            if done:
                break
        eval_reward.append(total_reward)
    return np.mean(eval_reward)


# 创建飞行器环境
env = make_env("Quadrotor", task="hovering_control")
# 将神经网络输出（取值范围为[-1, 1]）映射到对应的实际动作取值范围内

env.reset()
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

# 使用parl框架搭建Agent：QuadrotorModel, DDPG, QuadrotorAgent三者嵌套
model = QuadrotorModel(act_dim)
algorithm = DDPG(
    model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
agent = QuadrotorAgent(algorithm, obs_dim, act_dim)

# parl库也为DDPG算法内置了ReplayMemory，可直接从 parl.utils 引入使用
rpm = ReplayMemory(int(MEMORY_SIZE), obs_dim, act_dim)

test_flag = 0
total_steps = 0
while total_steps < TRAIN_TOTAL_STEPS:
    train_reward, steps = run_episode(env, agent, rpm)
    total_steps += steps
    #logger.info('Steps: {} Reward: {}'.format(total_steps, train_reward))

    if total_steps // TEST_EVERY_STEPS >= test_flag:
        while total_steps // TEST_EVERY_STEPS >= test_flag:
            test_flag += 1

        evaluate_reward = evaluate(env, agent)
        logger.info('Steps {}, Test reward: {}'.format(total_steps,
                                                       evaluate_reward))

        # 保存模型
        ckpt = 'model_dir/steps_{}.ckpt'.format(total_steps)
        agent.save(ckpt)
