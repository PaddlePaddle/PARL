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

import gym
import argparse
import numpy as np
from parl.utils import logger, summary, ReplayMemory
from parl.env.continuous_wrappers import ActionMappingWrapper
from mujoco_model import MujocoModel
from mujoco_agent import MujocoAgent
from parl.algorithms import TD3

MAX_EPISODES = 5000
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
GAMMA = 0.99
TAU = 0.005
MEMORY_SIZE = int(1e6)
WARMUP_SIZE = 1e4
BATCH_SIZE = 256
ENV_SEED = 1
EXPL_NOISE = 0.1  # Std of Gaussian exploration noise


def run_train_episode(env, agent, rpm):
    act_dim = env.action_space.shape[0]
    obs = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        steps += 1

        if rpm.size() < WARMUP_SIZE:
            action = np.random.uniform(-1, 1, size=act_dim)
        else:
            action = agent.sample(np.array(obs))

        next_obs, reward, done, info = env.step(action)
        done = float(done) if steps < env._max_episode_steps else 0
        rpm.append(obs, action, reward, next_obs, done)

        obs = next_obs
        total_reward += reward

        if rpm.size() > WARMUP_SIZE:
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(
                BATCH_SIZE)
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                        batch_terminal)

    return total_reward, steps


def run_evaluate_episode(env, agent):
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.predict(np.array(obs))
        obs, reward, done, _ = env.step(action)
        total_reward += reward
    return total_reward


def main():
    env = gym.make(args.env)
    env.seed(ENV_SEED)
    env = ActionMappingWrapper(env)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model = MujocoModel(obs_dim, act_dim)
    algorithm = TD3(
        model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    agent = MujocoAgent(algorithm, obs_dim, act_dim, expl_noise=EXPL_NOISE)

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
        default=int(3e6),
        help='maximum training steps')
    parser.add_argument(
        '--test_every_steps',
        type=int,
        default=int(1e4),
        help='the step interval between two consecutive evaluations')

    args = parser.parse_args()

    main()
