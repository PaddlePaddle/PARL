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
from remote_env import CarlaEnv
from parl.utils import logger, tensorboard, ReplayMemory
from parl.env.continuous_wrappers import ActionMappingWrapper
from carla_model import CarlaModel
from carla_agent import CarlaAgent
from parl.algorithms import SAC

WARMUP_STEPS = 2e3
EVAL_EVERY_STEPS = 1e3
EVAL_EPISODES = 3
MEMORY_SIZE = int(1e4)
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2  # determines the relative importance of entropy term against the reward
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
_max_episode_steps = 250


# Runs policy for 3 episodes by default and returns average reward
# A fixed seed is used for the eval environment
def run_evaluate_episodes(agent, env, eval_episodes):
    avg_reward = 0.
    for k in range(eval_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < _max_episode_steps:
            steps += 1
            action = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    return avg_reward


def main():
    logger.info("-----------------Carla_SAC-------------------")
    logger.info('Env: {}, Seed: {}'.format(args.env, args.seed))
    logger.info("---------------------------------------------")
    logger.set_dir('./{}_{}'.format(args.env, args.seed))

    # envs for training
    ports = [2021, 2023, 2025]
    parl.connect(args.localhost)
    env_list = [CarlaEnv(port=port) for port in ports]

    # env for eval
    params = {
        'obs_size': (160, 100),  # screen size of cv2 window
        'dt': 0.025,  # time interval between two frames
        'ego_vehicle_filter':
        'vehicle.lincoln*',  # filter for defining ego vehicle
        'port': 2027,  # connection port
        'task_mode':
        'Lane',  # mode of the task, [random, roundabout (only for Town03)]
        'code_mode': 'test',
        'max_time_episode': 250,  # maximum timesteps per episode
        'desired_speed': 15,  # desired speed (m/s)
        'max_ego_spawn_times': 100,  # maximum times to spawn ego vehicle
    }
    env = gym.make('carla-v0', params=params)
    env.seed(args.seed)
    env = ActionMappingWrapper(env)

    obs_dim = env.state_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Initialize model, algorithm, agent, replay_memory
    model = CarlaModel(obs_dim, action_dim)
    algorithm = SAC(
        model,
        gamma=GAMMA,
        tau=TAU,
        alpha=ALPHA,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR)
    agent = CarlaAgent(algorithm)
    rpm = ReplayMemory(
        max_size=MEMORY_SIZE, obs_dim=obs_dim, act_dim=action_dim)

    total_steps = 0
    test_flag = 0
    obs_list = [env.reset() for env in env_list]
    obs_list = [obs.get() for obs in obs_list]
    obs_list = np.array(obs_list)
    done_list = [False] * len(env_list)
    episode_reward_list = [0] * len(env_list)
    episode_steps_list = [0] * len(env_list)

    while total_steps < args.max_timesteps:
        # Train episode
        if rpm.size() < WARMUP_STEPS:
            action_list = [
                np.random.uniform(-1, 1, size=action_dim)
                for _ in range(len(env_list))
            ]
        else:
            action_list = [agent.sample(obs) for obs in obs_list]
        return_list = [
            env_list[i].step(action_list[i]) for i in range(len(env_list))
        ]
        return_list = [return_.get() for return_ in return_list]
        return_list = np.array(return_list, dtype=object)

        done_list = return_list[:, 2]
        # Store data in replay memory
        for i in range(len(return_list)):
            next_obs, reward, done, _ = return_list[i, 0], return_list[
                i, 1], return_list[i, 2], return_list[i, 3]
            rpm.append(obs_list[i], action_list[i], reward, next_obs, done)

            obs_list[i] = return_list[i, 0]

            total_steps += 1
            episode_steps_list[i] += 1
            episode_reward_list[i] += reward
            if done or episode_steps_list[i] >= _max_episode_steps:
                tensorboard.add_scalar('train/episode_reward_env{}'.format(i),
                                       episode_reward_list[i], total_steps)
                logger.info('Total Steps: {} Reward: {}'.format(
                    total_steps, episode_reward_list[i]))

                # reset env if done
                episode_steps_list[i] = 0
                episode_reward_list[i] = 0
                obs_list_i = env_list[i].reset()
                obs_list[i] = obs_list_i.get()
                obs_list[i] = np.array(obs_list[i])

        # Train agent after collecting sufficient data
        if rpm.size() >= WARMUP_STEPS:
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(
                BATCH_SIZE)
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                        batch_terminal)

        # Save agent
        if total_steps > int(1e5) and total_steps % int(1e4) == 0:
            agent.save('./model/{}_model.ckpt'.format(total_steps))

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
        "--localhost",
        default='172.18.138.13:8765',
        help='localhost to provide carla environment')
    parser.add_argument("--env", default="carla-v0")
    parser.add_argument("--task_mode", default='Lane', help='mode of the task')
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help='sets carla env seed for evaluation')
    parser.add_argument(
        "--max_timesteps",
        default=5e5,
        type=int,
        help='max time steps to run environment')
    args = parser.parse_args()

    main()
