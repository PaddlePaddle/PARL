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
import gym
import argparse
from parl.utils import logger, tensorboard, ReplayMemory
from parl.env.continuous_wrappers import ActionMappingWrapper
from mujoco_model import MujocoModel
from mujoco_agent import MujocoAgent
from parl.algorithms import SAC

WARMUP_STEPS = 1e4
EVAL_EVERY_STEPS = 5e3
EVAL_EPISODES = 5
MEMORY_SIZE = int(1e6)
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2  # determines the relative importance of entropy term against the reward
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4


# Run episode for training
def run_train_episode(agent, env, rpm):
    obs = env.reset()
    done = False
    episode_reward, episode_steps = 0, 0
    while not done:
        episode_steps += 1
        # Select action randomly or according to policy
        if rpm.size() < WARMUP_STEPS:
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
        if rpm.size() >= WARMUP_STEPS:
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(
                BATCH_SIZE)
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                        batch_terminal)

    return episode_reward, episode_steps


# Runs policy for 5 episodes by default and returns average reward
# A fixed seed is used for the eval environment
def run_evaluate_episodes(agent, env, eval_episodes):
    avg_reward = 0.
    for _ in range(eval_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.predict(np.array(state))
            state, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    return avg_reward


def main():
    logger.info("------------------- SAC ---------------------")
    logger.info('Env: {}, Seed: {}'.format(args.env, args.seed))
    logger.info("---------------------------------------------")

    env = gym.make(args.env)
    env.seed(args.seed)
    env = ActionMappingWrapper(env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Initialize model, algorithm, agent, replay_memory
    model = MujocoModel(state_dim, action_dim)
    algorithm = SAC(
        model,
        gamma=GAMMA,
        tau=TAU,
        alpha=ALPHA,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR)
    agent = MujocoAgent(algorithm, state_dim, action_dim)
    rpm = ReplayMemory(
        max_size=MEMORY_SIZE, obs_dim=state_dim, act_dim=action_dim)

    total_steps = 0
    test_flag = 0

    while total_steps < args.max_timesteps:
        # Train episode
        episode_reward, episode_steps = run_train_episode(agent, env, rpm)
        total_steps += episode_steps

        tensorboard.add_scalar('train/episode_reward', episode_reward,
                               total_steps)
        logger.info('Total Steps: {} Reward: {}'.format(
            total_steps, episode_reward))

        # Evaluate episode
        if (total_steps + 1) // EVAL_EVERY_STEPS >= test_flag:
            while (total_steps + 1) // EVAL_EVERY_STEPS >= test_flag:
                test_flag += 1
            avg_reward = run_evaluate_episodes(agent, env, EVAL_EPISODES)
            tensorboard.add_scalar('eval/episode_reward', avg_reward,
                                   total_steps)
            logger.info('Evaluation over: {} episodes, Reward: {}'.format(
                EVAL_EPISODES, avg_reward))


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
        "--max_timesteps",
        default=5e6,
        type=int,
        help='Max time steps to run environment')
    args = parser.parse_args()

    main()
