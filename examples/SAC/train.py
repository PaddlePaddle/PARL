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

# Refer to https://github.com/pranz24/pytorch-soft-actor-critic

import argparse
import gym
import numpy as np
import time
import parl
from mujoco_agent import MujocoAgent
from mujoco_model import ActorModel, CriticModel
from parl.utils import logger, summary, action_mapping, ReplayMemory

ACTOR_LR = 1e-3
CRITIC_LR = 1e-3
GAMMA = 0.99
TAU = 0.005
MEMORY_SIZE = int(1e6)
WARMUP_SIZE = 1e4
BATCH_SIZE = 256
ENV_SEED = 1


def run_train_episode(env, agent, rpm):
    obs = env.reset()
    total_reward = 0
    steps = 0
    while True:
        steps += 1
        batch_obs = np.expand_dims(obs, axis=0)

        if rpm.size() < WARMUP_SIZE:
            action = env.action_space.sample()
        else:
            action = agent.sample(batch_obs.astype('float32'))
            action = np.squeeze(action)

        next_obs, reward, done, info = env.step(action)

        rpm.append(obs, action, reward, next_obs, done)

        if rpm.size() > WARMUP_SIZE:
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(
                BATCH_SIZE)
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                        batch_terminal)

        obs = next_obs
        total_reward += reward

        if done:
            break
    return total_reward, steps


def run_evaluate_episode(env, agent):
    obs = env.reset()
    total_reward = 0
    while True:
        batch_obs = np.expand_dims(obs, axis=0)
        action = agent.predict(batch_obs.astype('float32'))
        action = np.squeeze(action)

        next_obs, reward, done, info = env.step(action)

        obs = next_obs
        total_reward += reward

        if done:
            break
    return total_reward


def main():
    env = gym.make(args.env)
    env.seed(ENV_SEED)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    actor = ActorModel(act_dim)
    critic = CriticModel()
    algorithm = parl.algorithms.SAC(
        actor,
        critic,
        max_action=max_action,
        gamma=GAMMA,
        tau=TAU,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR)
    agent = MujocoAgent(algorithm, obs_dim, act_dim)

    rpm = ReplayMemory(MEMORY_SIZE, obs_dim, act_dim)

    test_flag = 0
    total_steps = 0
    while total_steps < args.train_total_steps:
        train_reward, steps = run_train_episode(env, agent, rpm)
        total_steps += steps
        logger.info('Steps: {} Reward: {}'.format(total_steps, train_reward))
        summary.add_scalar('train/episode_reward', train_reward, total_steps)

        if total_steps // args.test_every_steps >= test_flag:
            while total_steps // args.test_every_steps >= test_flag:
                test_flag += 1
            evaluate_reward = run_evaluate_episode(env, agent)
            logger.info('Steps {}, Evaluate reward: {}'.format(
                total_steps, evaluate_reward))
            summary.add_scalar('eval/episode_reward', evaluate_reward,
                               total_steps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env', help='Mujoco environment name', default='HalfCheetah-v2')
    parser.add_argument(
        '--train_total_steps',
        type=int,
        default=int(1e6),
        help='maximum training steps')
    parser.add_argument(
        '--test_every_steps',
        type=int,
        default=int(1e4),
        help='the step interval between two consecutive evaluations')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.2,
        help='Temperature parameter α determines the relative importance of the \
        entropy term against the reward (default: 0.2)')

    args = parser.parse_args()

    main()
