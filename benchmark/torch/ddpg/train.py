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
from parl.algorithms import DDPG


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(agent, env_name, seed, eval_episodes):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = agent.predict(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    return avg_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="DDPG")  # Policy name
    parser.add_argument(
        "--env", default="HalfCheetah-v1")  # OpenAI gym environment name
    parser.add_argument(
        "--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument(
        "--start_timesteps", default=1e4,
        type=int)  # Time steps initial random policy is used
    parser.add_argument(
        "--eval_freq", default=5e3,
        type=int)  # How often (time steps) we evaluate
    parser.add_argument(
        "--eval_episodes", default=10,
        type=int)  # How many episodes during evaluation
    parser.add_argument(
        "--max_timesteps", default=1e6,
        type=int)  # Max time steps to run environment
    parser.add_argument(
        "--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument(
        "--batch_size", default=100, type=int)  # Batch size for learning
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument(
        "--actor_lr", default=1e-3)  # Learning rate of actor network
    parser.add_argument(
        "--critic_lr", default=1e-3)  # Learning rate of critic network
    parser.add_argument(
        "--policy_freq", default=1,
        type=int)  # Frequency to train actor and update params
    args = parser.parse_args()

    print("---------------------------------------")
    logger.info('Policy: {}, Env: {}, Seed: {}'.format(args.policy, args.env,
                                                       args.seed))
    print("---------------------------------------")

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize model, algorithm, agent, replay_memory
    model = MujocoModel(state_dim, action_dim, max_action)
    algorithm = DDPG(
        model,
        max_action,
        discount=args.discount,
        tau=args.tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        policy_freq=args.policy_freq)
    agent = MujocoAgent(algorithm, state_dim, action_dim)
    replay_memory = ReplayMemory(
        max_size=int(1e6), obs_dim=state_dim, act_dim=action_dim)

    obs, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (agent.predict(np.array(obs)) + np.random.normal(
                0, max_action * args.expl_noise, size=action_dim)).clip(
                    -max_action, max_action)

        # Perform action
        next_obs, reward, done, _ = env.step(action)
        terminal = float(
            done) if episode_timesteps < env._max_episode_steps else 0
        terminal = 1. - terminal

        # Store data in replay memory
        replay_memory.append(obs, action, reward, next_obs, terminal)

        obs = next_obs
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = replay_memory.sample_batch(
                args.batch_size)
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                        batch_terminal)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            logger.info('Episode: {} Steps: {} Reward: {}'.format(
                episode_num + 1, t + 1, episode_reward))
            tensorboard.add_scalar('train/episode_reward', episode_reward, t)

            # Reset environment
            obs, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            avg_reward = eval_policy(agent, args.env, args.seed,
                                     args.eval_episodes)
            logger.info('Evaluation over: {} episodes, Reward: {}'.format(
                args.eval_episodes, avg_reward))
            tensorboard.add_scalar('eval/episode_reward', avg_reward, t)
