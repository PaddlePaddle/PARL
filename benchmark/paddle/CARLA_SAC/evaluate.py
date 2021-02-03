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
from parl.utils import logger, tensorboard
from parl.env.continuous_wrappers import ActionMappingWrapper
from carla_model import CarlaModel
from carla_agent import CarlaAgent
from sac import SAC
# from parl.algorithms import SAC # parl >= 1.4.2

EVAL_EPISODES = 3
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2  # determines the relative importance of entropy term against the reward
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
MAX_EPISODE_STEPS = 250


def run_evaluate_episodes(agent, eval_env):
    episode_reward = 0.
    obs, _ = eval_env.reset()
    done = False
    steps = 0
    while not done and steps < MAX_EPISODE_STEPS:
        steps += 1
        action = agent.predict(obs)
        obs, reward, done, _ = eval_env.step(action)
        episode_reward += reward
    return episode_reward


def main():
    logger.info("-----------------Carla_SAC-------------------")
    logger.info('Env: {}, Seed: {}'.format(args.env, args.seed))
    logger.info("---------------------------------------------")
    logger.set_dir('./{}_eval_{}'.format(args.env, args.seed))

    # env for eval
    params = {
        'obs_size': (160, 100),  # screen size of cv2 window
        'dt': 0.025,  # time interval between two frames
        'ego_vehicle_filter':
        'vehicle.lincoln*',  # filter for defining ego vehicle
        'port': 2027,  # CARLA service's port
        'task_mode':
        'Lane',  # mode of the task, [random, roundabout (only for Town03)]
        'code_mode': 'test',
        'max_time_episode': MAX_EPISODE_STEPS,  # maximum timesteps per episode
        'desired_speed': 15,  # desired speed (m/s)
        'max_ego_spawn_times': 100,  # maximum times to spawn ego vehicle
    }
    eval_env = gym.make('carla-v0', params=params)
    eval_env.seed(args.seed)
    eval_env = ActionMappingWrapper(eval_env)

    obs_dim = eval_env.state_space.shape[0]
    action_dim = eval_env.action_space.shape[0]

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
    for episode in range(args.evaluate_episodes):
        episode_reward = run_evaluate_episodes(agent, eval_env)
        tensorboard.add_scalar('eval/episode_reward', episode_reward, episode)
        logger.info('Evaluation episode reward: {}'.format(episode_reward))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="carla-v0")
    parser.add_argument("--task_mode", default='Lane', help='mode of the task')
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help='sets carla env seed for evaluation')
    parser.add_argument(
        "--evaluate_episodes",
        default=1e4,
        type=int,
        help='max time steps to run environment')
    args = parser.parse_args()

    main()
