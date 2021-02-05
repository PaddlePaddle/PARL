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

import gym
import numpy as np
import parl
import argparse
import carla
import gym_carla
from remote_env import LocalEnv
from parl.utils import logger, tensorboard
from parl.env.continuous_wrappers import ActionMappingWrapper
from carla_model import CarlaModel
from carla_agent import CarlaAgent
from sac import SAC
# from parl.algorithms import SAC # parl >= 1.4.2
from env_config import EnvConfig

EVAL_EPISODES = 3
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2  # determines the relative importance of entropy term against the reward
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4


def run_episode(agent, env):
    episode_reward = 0.
    obs = env.reset()
    done = False
    steps = 0
    while not done and steps < env._max_episode_steps:
        steps += 1
        action = agent.predict(obs)
        obs, reward, done, _ = env.step(action)
        episode_reward += reward
    return episode_reward


def main():
    logger.info("-----------------Carla_SAC-------------------")
    logger.set_dir('./{}_eval'.format(args.env))

    # env for eval
    eval_env_params = EnvConfig['eval_env_params']
    eval_env = LocalEnv(args.env, eval_env_params)

    obs_dim = eval_env.obs_dim
    action_dim = eval_env.action_dim

    # Initialize model, algorithm, agent
    model = CarlaModel(obs_dim, action_dim)
    algorithm = SAC(
        model,
        gamma=GAMMA,
        tau=TAU,
        alpha=ALPHA,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR)
    agent = CarlaAgent(algorithm)
    agent.restore('./model.ckpt')

    # Evaluate episode
    for episode in range(args.eval_episodes):
        episode_reward = run_episode(agent, eval_env)
        tensorboard.add_scalar('eval/episode_reward', episode_reward, episode)
        logger.info('Evaluation episode reward: {}'.format(episode_reward))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="carla-v0")
    parser.add_argument("--task_mode", default='Lane', help='mode of the task')
    parser.add_argument(
        "--eval_episodes",
        default=10,
        type=int,
        help='max time steps to run environment')
    args = parser.parse_args()

    main()
