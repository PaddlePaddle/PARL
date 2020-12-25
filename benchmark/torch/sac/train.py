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

import numpy as np
import random
import torch
import gym
import argparse
from parl.utils import logger, tensorboard, ReplayMemory
from mujoco_model import MujocoModel
from mujoco_agent import MujocoAgent
from parl.algorithms import SAC


# Run episode for training
def run_train_episode(agent, env, rpm):
    obs, done = env.reset(), False
    episode_reward, episode_steps = 0, 0
    while not done:
        episode_steps += 1
        # Select action randomly or according to policy
        if rpm.size() < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = agent.predict(np.array(obs))

        # Perform action
        next_obs, reward, done, _ = env.step(action)
        terminal = float(done) if episode_steps < env._max_episode_steps else 0
        terminal = 1. - terminal

        # Store data in replay memory
        rpm.append(obs, action, reward, next_obs, terminal)

        obs = next_obs
        episode_reward += reward

        # Train agent after collecting sufficient data
        if rpm.size() >= args.start_timesteps:
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(
                args.batch_size)
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                        batch_terminal)

    return episode_reward, episode_steps


# Runs policy for 5 episodes by default and returns average reward
# A fixed seed is used for the eval environment
def run_evaluate_episodes(agent, eval_env, eval_episodes=5):
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = agent.predict(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    return avg_reward


def main():
    logger.info("------------------- SAC ---------------------")
    logger.info('Env: {}, Seed: {}'.format(args.env, args.seed))
    logger.info("---------------------------------------------")

    env = gym.make(args.env)
    eval_env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    eval_env.seed(args.seed + 100)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize model, algorithm, agent, replay_memory
    model = MujocoModel(state_dim, action_dim)
    algorithm = SAC(
        model,
        max_action,
        discount=args.discount,
        tau=args.tau,
        alpha=args.alpha,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        policy_freq=args.policy_freq,
        automatic_entropy_tuning=False,
        entropy_lr=args.entropy_lr,
        action_dim=action_dim)
    agent = MujocoAgent(algorithm, state_dim, action_dim)
    rpm = ReplayMemory(
        max_size=int(1e6), obs_dim=state_dim, act_dim=action_dim)

    episode_num = 0
    total_steps = 0
    test_flag = 0

    while total_steps < args.max_timesteps:
        # Train episode
        episode_num += 1
        episode_reward, episode_steps = run_train_episode(agent, env, rpm)
        total_steps += episode_steps

        tensorboard.add_scalar('train/episode_reward', episode_reward,
                               total_steps)
        logger.info('Total Steps: {} Episode: {} Reward: {}'.format(
            total_steps, episode_num, episode_reward))

        # Evaluate episode
        if (total_steps + 1) // args.eval_freq >= test_flag:
            while (total_steps + 1) // args.eval_freq >= test_flag:
                test_flag += 1
            avg_reward = run_evaluate_episodes(agent, eval_env,
                                               args.eval_episodes)
            tensorboard.add_scalar('eval/episode_reward', avg_reward,
                                   total_steps)
            logger.info('Evaluation over: {} episodes, Reward: {}'.format(
                args.eval_episodes, avg_reward))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", default="HalfCheetah-v1", help='Mujoco gym environment name')
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help='Sets Gym, PyTorch and Numpy seeds')
    parser.add_argument(
        "--start_timesteps",
        default=1e4,
        type=int,
        help='Time steps initial, random policy is used')
    parser.add_argument(
        "--eval_freq", default=5e3, type=int, help='How often to evaluate')
    parser.add_argument(
        "--eval_episodes",
        default=5,
        type=int,
        help='How many episodes during evaluation')
    parser.add_argument(
        "--max_timesteps",
        default=1e6,
        type=int,
        help='Max time steps to run environment')
    parser.add_argument(
        "--alpha", default=0.2, help='Temperature parameter'
    )  # determines the relative importance of entropy term against the reward
    parser.add_argument(
        "--batch_size", default=256, type=int, help='Batch size for learning')
    parser.add_argument("--discount", default=0.99, help='Discount factor')
    parser.add_argument(
        "--tau", default=5e-3, help='Target network update rate')
    parser.add_argument(
        "--actor_lr", default=3e-4, help='Learning rate of actor network')
    parser.add_argument(
        "--critic_lr", default=3e-4, help='Learning rate of critic network')
    parser.add_argument(
        "--policy_freq",
        default=1,
        type=int,
        help=
        'Frequency to train actor(& adjust alpha if necessary) and update params'
    )
    parser.add_argument(
        "--automatic_entropy_tuning",
        default=False,
        type=bool,
        help='Whether or not to adjust alpha automatically ')
    parser.add_argument(
        "--entropy_lr", default=3e-4, help='Learning rate of entropy')

    args = parser.parse_args()

    main()