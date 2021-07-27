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
import os
import torch
import parl
import gym

import numpy as np
from tqdm import tqdm
from parl.utils import summary, logger
from parl.algorithms import DQN, DDQN
from parl.env.atari_wrappers import wrap_deepmind
from agent import AtariAgent
from model import AtariModel
from replay_memory import ReplayMemory, Experience

# env params
IMAGE_SIZE = (84, 84)
CONTEXT_LEN = 4
FRAME_SKIP = 4

# model params
UPDATE_TARGET_STEP = 2500
MEMORY_SIZE = 1000000
GAMMA = 0.99
LR_START = 0.0003  # LR start
TOTAL_STEP = 1000000
MEMORY_WARMUP_SIZE = 50000
UPDATE_FREQ = 4
BATCH_SIZE = 32

# eval params
EVAL_RENDER = False


# train an episode
def run_train_episode(agent, env, rpm):
    total_reward = 0
    obs = env.reset()
    step = 0
    loss_lst = []

    while True:
        step += 1
        context = rpm.recent_obs()
        context.append(obs)
        context = np.stack(context, axis=0)

        action = agent.sample(context)
        next_obs, reward, done, _ = env.step(action)
        rpm.append(Experience(obs, action, reward, done))

        # train model
        if (rpm.size() > MEMORY_WARMUP_SIZE) and (step % UPDATE_FREQ == 0):
            # s,a,r,s',done
            (batch_all_obs, batch_action, batch_reward,
             batch_done) = rpm.sample_batch(BATCH_SIZE)
            batch_obs = batch_all_obs[:, :CONTEXT_LEN, :, :]
            batch_next_obs = batch_all_obs[:, 1:, :, :]

            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs, batch_done)
            loss_lst.append(train_loss)

        total_reward += reward
        obs = next_obs
        if done:
            break
    return total_reward, step, np.mean(loss_lst)


def run_evaluate_episodes(agent, env):
    obs = env.reset()
    while not env.get_real_done():
        action = agent.predict(obs)
        obs, _, done, _ = env.step(action)
        if EVAL_RENDER:
            env.render()
        if done:
            obs = env.reset()
    return np.mean(env.get_eval_rewards())


def main():
    env = gym.make(args.env)
    env = wrap_deepmind(
        env, dim=IMAGE_SIZE[0], framestack=False, obs_format='NCHW')
    test_env = gym.make(args.env)
    test_env = wrap_deepmind(
        test_env, dim=IMAGE_SIZE[0], obs_format='NCHW', test=True)

    # set seed
    torch.manual_seed(args.train_seed)
    torch.cuda.manual_seed_all(args.train_seed)
    env.seed(args.train_seed)
    test_env.seed(args.test_seed)

    rpm = ReplayMemory(MEMORY_SIZE, IMAGE_SIZE, CONTEXT_LEN)
    act_dim = env.action_space.n

    # get model
    model = AtariModel(act_dim, args.dueling)

    # get algorithm
    algo_name = args.algo
    if algo_name == 'DQN':
        alg = DQN(model, gamma=GAMMA, lr=LR_START)
    else:
        alg = DDQN(model, gamma=GAMMA, lr=LR_START)

    # get agent
    agent = AtariAgent(
        alg,
        act_dim=act_dim,
        total_step=TOTAL_STEP,
        update_target_step=UPDATE_TARGET_STEP,
        start_lr=LR_START)

    # start training, memory warm up
    with tqdm(
            total=MEMORY_WARMUP_SIZE, desc='[Replay Memory Warm Up]') as pbar:
        while rpm.size() < MEMORY_WARMUP_SIZE:
            total_reward, steps, _ = run_train_episode(agent, env, rpm)
            pbar.update(steps)

    test_flag = 0
    train_total_steps = args.train_total_steps
    pbar = tqdm(total=train_total_steps)
    cum_steps = 0  # this is the current timestep
    while cum_steps < train_total_steps:
        # start epoch
        total_reward, steps, loss = run_train_episode(agent, env, rpm)
        cum_steps += steps

        pbar.set_description('[train]exploration, learning rate:{}, {}'.format(
            agent.curr_ep, agent.alg.optimizer.param_groups[0]['lr']))
        summary.add_scalar('{}/training_rewards'.format(algo_name),
                           total_reward, cum_steps)
        summary.add_scalar('{}/loss'.format(algo_name), loss,
                           cum_steps)  # mean of total loss
        summary.add_scalar('{}/exploration'.format(algo_name), agent.curr_ep,
                           cum_steps)

        pbar.update(steps)

        # perform evaluation
        if cum_steps // args.test_every_steps >= test_flag:
            while cum_steps // args.test_every_steps >= test_flag:
                test_flag += 1

            pbar.write("testing")
            eval_rewards_mean = run_evaluate_episodes(agent, test_env)

            logger.info(
                "eval_agent done, (steps, eval_reward): ({}, {})".format(
                    cum_steps, eval_rewards_mean))

            summary.add_scalar('{}/mean_validation_rewards'.format(algo_name),
                               eval_rewards_mean, cum_steps)

    pbar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env', help='name of the atari env', default='PongNoFrameskip-v4')
    parser.add_argument(
        '--algo',
        default='DQN',
        type=str,
        help='DQN/DDQN, represent DQN, double DQN respectively')
    parser.add_argument(
        '--dueling',
        default=False,
        type=bool,
        help=
        'if True, represent dueling DQN or dueling DDQN, else ord DQN or DDQN')
    parser.add_argument(
        '--train_total_steps',
        type=int,
        default=int(1e7),
        help='maximum environmental steps of games')
    parser.add_argument(
        '--test_every_steps',
        type=int,
        default=100000,
        help='the step interval between two consecutive evaluations')
    parser.add_argument(
        '--train_seed',
        type=int,
        default=66166616,
        help='set the random seed for training environment')
    parser.add_argument(
        '--test_seed',
        type=int,
        default=666616,
        help='set the random seed for test and eval environment')

    args = parser.parse_args()
    main()
