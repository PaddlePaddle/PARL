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

import argparse
import gym
import d4rl
from parl.utils import logger, tensorboard, ReplayMemory
from parl.env.continuous_wrappers import ActionMappingWrapper
from mujoco_model import MujocoModel
from mujoco_agent import MujocoAgent
from parl.algorithms import CQL

POLICY_EVAL_START = 40000  # Defaulted to 20000 (40000 or 10000 work similarly)
LAGRANGE_THRESH = 10  # but paper: 10.0 in mujoco,  5.0 in Franka kitchen and Adroit domains
MIN_Q_VERSION = 3  # min_q_version = 3 (CQL(H)), = 2 (CQL(rho)), will be set to <0 in cql if not using lagrange
MIN_Q_WEIGHT = 5.0  # the value of alpha in critic loss, suggest 5.0 or 10.0 if not using lagrange
ALPHA = 1.0  # the value of alpha(temperature parameter) in actor loss, determines the relative importance of entropy term against the reward
EVAL_EPISODES = 5
MEMORY_SIZE = int(2e6)
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
ACTOR_LR = 1e-4
CRITIC_LR = 3e-4


# Runs policy for 5 episodes by default and returns average reward
# A fixed seed is used for the eval environment
def run_evaluate_episodes(agent, env, eval_episodes):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    return avg_reward


def load_hdf5(dataset, rpm):
    rpm.obs = dataset['observations']
    rpm.next_obs = dataset['next_observations']
    rpm.action = dataset['actions']
    rpm.reward = dataset['rewards']
    rpm.terminal = dataset['terminals']
    rpm._curr_size = dataset['terminals'].shape[0]

    logger.info("Dataset Info: ")
    for key in dataset:
        logger.info('key: {},\tshape: {},\tdtype: {}'.format(
            key, dataset[key].shape, dataset[key].dtype))
    logger.info('Number of terminals on: {}'.format(rpm.terminal.sum()))


def main():
    logger.info("------------------- CQL ---------------------")
    logger.info('Env: {}, Seed: {}'.format(args.env, args.seed))
    logger.info("---------------------------------------------")

    env = gym.make(args.env)
    env.seed(args.seed)
    env = ActionMappingWrapper(env)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Initialize model, algorithm, agent
    model = MujocoModel(obs_dim, action_dim)
    algorithm = CQL(
        model,
        gamma=GAMMA,
        tau=TAU,
        alpha=ALPHA,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        policy_eval_start=POLICY_EVAL_START,
        with_automatic_entropy_tuning=args.with_automatic_entropy_tuning,
        with_lagrange=args.with_lagrange,
        lagrange_thresh=LAGRANGE_THRESH,
        min_q_version=MIN_Q_VERSION,
        min_q_weight=MIN_Q_WEIGHT)
    agent = MujocoAgent(algorithm)

    # Initialize offline data
    rpm = ReplayMemory(
        max_size=MEMORY_SIZE, obs_dim=obs_dim, act_dim=action_dim)
    load_hdf5(d4rl.qlearning_dataset(env), rpm)

    total_steps = 0
    test_flag = 0
    while total_steps < args.train_total_steps:
        # Train steps
        batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(
            BATCH_SIZE)
        agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                    batch_terminal)
        total_steps += 1

        # Evaluate episode
        if (total_steps + 1) // args.test_every_steps >= test_flag:
            while (total_steps + 1) // args.test_every_steps >= test_flag:
                test_flag += 1
            avg_reward = run_evaluate_episodes(agent, env, EVAL_EPISODES)
            tensorboard.add_scalar('eval/episode_reward', avg_reward,
                                   total_steps)
            logger.info('Evaluation: total_steps {}, Reward: {}'.format(
                total_steps, avg_reward))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        default="halfcheetah-medium-expert-v0",
        help='Mujoco gym environment name in d4rl')
    parser.add_argument(
        "--seed",
        default=10,
        type=int,
        help='Sets Gym, PyTorch and Numpy seeds')
    parser.add_argument(
        "--train_total_steps",
        default=1e6,
        type=int,
        help='Max time steps to run environment')
    parser.add_argument(
        '--test_every_steps',
        type=int,
        default=int(1e4),
        help='The step interval between two consecutive evaluations')
    parser.add_argument(
        '--with_automatic_entropy_tuning',
        dest='with_automatic_entropy_tuning',
        action='store_true',
        default=False)
    parser.add_argument(
        '--with_lagrange',
        dest='with_lagrange',
        action='store_true',
        default=False)

    args = parser.parse_args()
    logger.info(args)

    main()
