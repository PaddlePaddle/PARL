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
import argparse
import numpy as np
import paddle
from paddle.static import InputSpec
from parl.utils import logger, summary, ReplayMemory
from parl.env.continuous_wrappers import ActionMappingWrapper
from mujoco_model import MujocoModel
from mujoco_agent import MujocoAgent
from parl.algorithms import DDPG

WARMUP_STEPS = 1e4
EVAL_EPISODES = 5
MEMORY_SIZE = int(1e6)
BATCH_SIZE = 100
GAMMA = 0.99
TAU = 0.005
ACTOR_LR = 1e-3
CRITIC_LR = 1e-3
EXPL_NOISE = 0.1  # Std of Gaussian exploration noise


# Run episode for training
def run_train_episode(agent, env, rpm):
    action_dim = env.action_space.shape[0]
    obs = env.reset()
    done = False
    episode_reward, episode_steps = 0, 0

    while not done:
        episode_steps += 1
        # Select action randomly or according to policy
        if rpm.size() < WARMUP_STEPS:
            action = np.random.uniform(-1, 1, size=action_dim)
        else:
            action = agent.sample(obs)

        # Perform action
        next_obs, reward, done, _ = env.step(action)
        terminal = float(done) if episode_steps < env._max_episode_steps else 0

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
        obs = env.reset()
        done = False
        while not done:
            action = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    return avg_reward


# Runs policy for 5 episodes by default and returns average reward
# A fixed seed is used for the infer environment
def run_inference_episodes(path, env, inference_episodes):
    inference_model = paddle.jit.load(path)
    avg_reward = 0
    for _ in range(inference_episodes):
        obs = env.reset()
        done = False
        while not done:
            obs = paddle.to_tensor(obs, dtype='float32')
            action = inference_model(obs).cpu().numpy()[0]
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= inference_episodes
    return avg_reward


def main():
    logger.info("------------------ DDPG ---------------------")
    logger.info('Env: {}, Seed: {}'.format(args.env, args.seed))
    logger.info("---------------------------------------------")
    logger.set_dir('./{}_{}'.format(args.env, args.seed))

    env = gym.make(args.env)
    env.seed(args.seed)
    env = ActionMappingWrapper(env)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Initialize model, algorithm, agent, replay_memory
    model = MujocoModel(obs_dim, action_dim)
    algorithm = DDPG(
        model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    agent = MujocoAgent(algorithm, action_dim, expl_noise=EXPL_NOISE)
    rpm = ReplayMemory(
        max_size=MEMORY_SIZE, obs_dim=obs_dim, act_dim=action_dim)

    total_steps = 0
    test_flag = 0
    while total_steps < args.train_total_steps:
        # Train episode
        episode_reward, episode_steps = run_train_episode(agent, env, rpm)
        total_steps += episode_steps

        summary.add_scalar('train/episode_reward', episode_reward, total_steps)
        logger.info('Total Steps: {} Reward: {}'.format(
            total_steps, episode_reward))

        # Evaluate episode
        if (total_steps + 1) // args.test_every_steps >= test_flag:
            while (total_steps + 1) // args.test_every_steps >= test_flag:
                test_flag += 1
            avg_reward = run_evaluate_episodes(agent, env, EVAL_EPISODES)
            summary.add_scalar('eval/episode_reward', avg_reward, total_steps)
            logger.info('Evaluation over: {} episodes, Reward: {}'.format(
                EVAL_EPISODES, avg_reward))

    # save the model and parameters of policy network for inference
    save_inference_path = './inference_model'
    input_spec = InputSpec(shape=[None, env.observation_space.shape[0]], dtype='float32')
    agent.save_inference_model(save_inference_path, input_spec, model.actor_model)
    inference_reward = run_inference_episodes(save_inference_path, env, inference_episodes=5)
    logger.info('Inference reward:{}'.format(inference_reward))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", default="HalfCheetah-v1", help='OpenAI gym environment name')
    parser.add_argument("--seed", default=0, type=int, help='Sets Gym seed')
    parser.add_argument(
        "--train_total_steps",
        default=5e6,
        type=int,
        help='Max time steps to run environment')
    parser.add_argument(
        '--test_every_steps',
        type=int,
        default=int(5e3),
        help='The step interval between two consecutive evaluations')
    args = parser.parse_args()

    main()
