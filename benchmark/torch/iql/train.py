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
import pickle
import argparse
import gym
from tqdm import trange
import d4rl
from parl.utils import logger, tensorboard
from replay_buffer import ReplayMemory
from mujoco_model import MujocoModel
from mujoco_agent import MujocoAgent
from parl.algorithms import IQL
import numpy as np
EVAL_EPISODES = 10
MEMORY_SIZE = int(2e6)
BATCH_SIZE = 256


# Runs policy for 5 episodes by default and returns average reward
# A fixed seed is used for the eval environment
def run_evaluate_episodes(agent, env, eval_episodes):
    eval_returns = []
    for _ in range(eval_episodes):
        avg_reward = 0.
        obs = env.reset()
        done = False
        while not done:
            action = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
        eval_returns.append(avg_reward)
    eval_returns = np.array(eval_returns)
    avg_reward = eval_returns.mean()
    return avg_reward, eval_returns


def main():
    logger.info("------------------- IQL ---------------------")
    logger.info('Env: {}, Seed: {}'.format(args.env, args.seed))
    logger.info("---------------------------------------------")

    env = gym.make(args.env)
    env.seed(args.seed)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Initialize model, algorithm, agent
    model = MujocoModel(obs_dim, action_dim)
    algorithm = IQL(model, max_steps=args.train_total_steps)
    agent = MujocoAgent(algorithm)

    # Initialize offline data
    rpm = ReplayMemory(
        max_size=MEMORY_SIZE, obs_dim=obs_dim, act_dim=action_dim)
    rpm.load_from_d4rl(d4rl.qlearning_dataset(env))

    result = []
    for total_steps in trange(args.train_total_steps):
        # Train steps
        batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(
            BATCH_SIZE)
        critic_loss, value_loss, actor_loss = agent.learn(
            batch_obs, batch_action, batch_reward, batch_next_obs,
            batch_terminal)
        tensorboard.add_scalar('train/critic_loss', critic_loss, total_steps)
        tensorboard.add_scalar('train/value_loss', value_loss, total_steps)
        tensorboard.add_scalar('train/actor_loss', actor_loss, total_steps)
        # Evaluate episode
        if total_steps % args.test_every_steps == 0:
            avg_reward, eval_rewards = run_evaluate_episodes(
                agent, env, EVAL_EPISODES)
            normalized_returns = d4rl.get_normalized_score(
                args.env, eval_rewards) * 100
            normalized_mean = normalized_returns.mean()
            tensorboard.add_scalar('eval/episode_reward', avg_reward,
                                   total_steps)
            tensorboard.add_scalar('eval/episode_normalized_reward',
                                   normalized_mean, total_steps)
            logger.info('Evaluation: total_steps {}, Reward: {}'.format(
                total_steps, avg_reward))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        default="hopper-medium-v2",
        help='Mujoco gym environment name in d4rl')
    parser.add_argument(
        "--seed",
        default=60,
        type=int,
        help='Sets Gym, PyTorch and Numpy seeds')
    parser.add_argument(
        "--train_total_steps",
        default=int(1e6),
        type=int,
        help='Max time steps to run environment')
    parser.add_argument(
        '--test_every_steps',
        type=int,
        default=int(5e3),
        help='The step interval between two consecutive evaluations')

    args = parser.parse_args()
    logger.info(args)

    main()
