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
import time
from mujoco_agent import MujocoAgent
from mujoco_model import MujocoModel
from parl.algorithms import DDPG
from parl.utils import logger
from replay_memory import ReplayMemory

MAX_EPISODES = 5000
TEST_EVERY_EPISODES = 50
MAX_STEPS_EACH_EPISODE = 1000
ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
GAMMA = 0.99
TAU = 0.001
MEMORY_SIZE = int(1e6)
MIN_LEARN_SIZE = 1e4
BATCH_SIZE = 128
REWARD_SCALE = 0.1
ENV_SEED = 1


def run_train_episode(env, agent, rpm, act_bound):
    obs = env.reset()
    total_reward = 0
    for j in range(MAX_STEPS_EACH_EPISODE):
        batch_obs = np.expand_dims(obs, axis=0)
        action = agent.predict(batch_obs.astype('float32'))
        action = np.squeeze(action)

        # Add exploration noise, and clip to [-1.0, 1.0]
        action = np.clip(
            np.random.normal(action, 1.0), -1.0, 1.0)
        action = action * act_bound

        next_obs, reward, done, info = env.step(action)

        rpm.append(obs, action, REWARD_SCALE * reward, next_obs, done)

        if rpm.size() > MIN_LEARN_SIZE:
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(
                BATCH_SIZE)
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                        batch_terminal)

        obs = next_obs
        total_reward += reward

        if done:
            break
    return total_reward


def run_evaluate_episode(env, agent, act_bound):
    obs = env.reset()
    total_reward = 0
    for j in range(MAX_STEPS_EACH_EPISODE):
        batch_obs = np.expand_dims(obs, axis=0)
        action = agent.predict(batch_obs.astype('float32'))
        action = np.squeeze(action)
        action = action * act_bound

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
    act_bound = env.action_space.high[0]

    model = MujocoModel(act_dim)
    algorithm = DDPG(
        model,
        hyperparas={
            'gamma': GAMMA,
            'tau': TAU,
            'actor_lr': ACTOR_LR,
            'critic_lr': CRITIC_LR
        })
    agent = MujocoAgent(algorithm, obs_dim, act_dim)

    rpm = ReplayMemory(MEMORY_SIZE, obs_dim, act_dim)

    for i in range(MAX_EPISODES):
        train_reward = run_train_episode(env, agent, rpm, act_bound)
        logger.info('Episode: {} Reward: {}'.format(i, train_reward))
        if (i + 1) % TEST_EVERY_EPISODES == 0:
            evaluate_reward = run_evaluate_episode(env, agent, act_bound)
            logger.info('Evaluate Reward: {}'.format(evaluate_reward))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env', help='Mujoco environment name', default='HalfCheetah-v2')
    args = parser.parse_args()
    main()
