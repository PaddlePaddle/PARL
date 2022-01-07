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

#-*- coding: utf-8 -*-

# 检查版本
import gym
import parl
import paddle
assert paddle.__version__ == "2.2.0", "[Version WARNING] please try `pip install paddlepaddle==2.2.0`"
assert parl.__version__ == "2.0.3", "[Version WARNING] please try `pip install parl==2.0.1`"
assert gym.__version__ == "0.18.0", "[Version WARNING] please try `pip install gym==0.18.0`"

#import gym
import argparse
import numpy as np
from parl.utils import logger, tensorboard, ReplayMemory
#from parl.env.continuous_wrappers import ActionMappingWrapper
from quadrotor_model import QuadrotorModel
from quadrotor_agent import QuadrotorAgent
from parl.algorithms import DDPG
from rlschool import make_env
import paddle


class ActionMappingWrapper:
    def __init__(self, env):
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.low_bound = self.env.action_space.low[0]
        self.high_bound = self.env.action_space.high[0]
        assert self.high_bound > self.low_bound

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, model_output_act):
        assert np.all(((model_output_act<=1.0 + 1e-3), (model_output_act>=-1.0 - 1e-3))), \
            'the action should be in range [-1, 1] !'
        assert self.high_bound > self.low_bound
        mapped_action = self.low_bound + (model_output_act - (-1.0)) * (
            (self.high_bound - self.low_bound) / 2.0)
        mapped_action = np.clip(mapped_action, self.low_bound, self.high_bound)
        return self.env.step(mapped_action)

    def render(self):
        self.env.render()


ACTOR_LR = 0.0002
CRITIC_LR = 0.001

GAMMA = 0.99
TAU = 0.005
MEMORY_SIZE = int(1e6)
WARMUP_STEPS = 1e4
REWARD_SCALE = 0.01

BATCH_SIZE = 256
EVAL_EPISODES = 5
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
        terminal = done

        # Store data in replay memory
        rpm.append(obs, action, REWARD_SCALE * reward, next_obs, terminal)
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
def run_evaluate_episodes(agent, env, eval_episodes, render=False):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
            if render:
                env.render()
    avg_reward /= eval_episodes
    return avg_reward


def main():

    env = make_env('Quadrotor', task='hovering_control')
    env = ActionMappingWrapper(env)
    env.reset()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Initialize model, algorithm, agent, replay_memory
    model = QuadrotorModel(obs_dim, action_dim)
    algorithm = DDPG(
        model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    agent = QuadrotorAgent(algorithm, action_dim, expl_noise=EXPL_NOISE)
    rpm = ReplayMemory(
        max_size=MEMORY_SIZE, obs_dim=obs_dim, act_dim=action_dim)

    total_steps = 0
    test_flag = 0
    while total_steps < args.train_total_steps:
        # Train episode
        episode_reward, episode_steps = run_train_episode(agent, env, rpm)
        total_steps += episode_steps

        tensorboard.add_scalar('train/episode_reward', episode_reward,
                               total_steps)
        logger.info('Total Steps: {} Reward: {}'.format(
            total_steps, episode_reward))

        # Evaluate episode
        if (total_steps + 1) // args.test_every_steps >= test_flag:
            while (total_steps + 1) // args.test_every_steps >= test_flag:
                test_flag += 1
            avg_reward = run_evaluate_episodes(
                agent, env, EVAL_EPISODES, render=False)
            tensorboard.add_scalar('eval/episode_reward', avg_reward,
                                   total_steps)
            logger.info('Evaluation over: {} episodes, Reward: {}'.format(
                EVAL_EPISODES, avg_reward))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_total_steps",
        default=5e6,
        type=int,
        help='Max time steps to run environment')
    parser.add_argument(
        '--test_every_steps',
        type=int,
        default=int(1e4),
        help='The step interval between two consecutive evaluations')
    args = parser.parse_args()

    main()
