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

from parl.utils import check_version_for_fluid  # requires parl >= 1.4.1
check_version_for_fluid()

import argparse
import os
import pickle
from collections import deque
from datetime import datetime

import gym
import numpy as np
import paddle.fluid as fluid
from tqdm import tqdm

import parl
from atari_agent import AtariAgent
from atari_model import AtariModel
from parl.utils import logger, summary
from per_alg import PrioritizedDoubleDQN, PrioritizedDQN
from proportional_per import ProportionalPER
from utils import get_player

MEMORY_SIZE = 1e6
MEMORY_WARMUP_SIZE = MEMORY_SIZE
IMAGE_SIZE = (84, 84)
CONTEXT_LEN = 4
FRAME_SKIP = 4
UPDATE_FREQ = 4
GAMMA = 0.99
LEARNING_RATE = 0.00025 / 4


def beta_adder(init_beta, step_size=0.0001):
    beta = init_beta
    step_size = step_size

    def adder():
        nonlocal beta, step_size
        beta += step_size
        return min(beta, 1)

    return adder


def process_transitions(transitions):
    transitions = np.array(transitions)
    batch_obs = np.stack(transitions[:, 0].copy())
    batch_act = transitions[:, 1].copy()
    batch_reward = transitions[:, 2].copy()
    batch_next_obs = np.expand_dims(np.stack(transitions[:, 3]), axis=1)
    batch_next_obs = np.concatenate([batch_obs, batch_next_obs],
                                    axis=1)[:, 1:, :, :].copy()
    batch_terminal = transitions[:, 4].copy()
    batch = (batch_obs, batch_act, batch_reward, batch_next_obs,
             batch_terminal)
    return batch


def run_episode(env, agent, per, mem=None, warmup=False, train=False):
    total_reward = 0
    all_cost = []
    traj = deque(maxlen=CONTEXT_LEN)
    obs = env.reset()
    for _ in range(CONTEXT_LEN - 1):
        traj.append(np.zeros(obs.shape))
    steps = 0
    if warmup:
        decay_exploration = False
    else:
        decay_exploration = True
    while True:
        steps += 1
        traj.append(obs)
        context = np.stack(traj, axis=0)
        action = agent.sample(context, decay_exploration=decay_exploration)
        next_obs, reward, terminal, _ = env.step(action)
        transition = [obs, action, reward, next_obs, terminal]
        if warmup:
            mem.append(transition)
        if train:
            per.store(transition)
            if steps % UPDATE_FREQ == 0:
                beta = get_beta()
                transitions, idxs, sample_weights = per.sample(beta=beta)
                batch = process_transitions(transitions)

                cost, delta = agent.learn(*batch, sample_weights)
                all_cost.append(cost)
                per.update(idxs, delta)

        total_reward += reward
        obs = next_obs
        if terminal:
            break

    return total_reward, steps, np.mean(all_cost)


def run_evaluate_episode(env, agent):
    obs = env.reset()
    total_reward = 0
    while True:
        action = agent.predict(obs)
        obs, reward, isOver, info = env.step(action)
        total_reward += reward
        if isOver:
            break
    return total_reward


def main():
    # Prepare environments
    env = get_player(
        args.rom, image_size=IMAGE_SIZE, train=True, frame_skip=FRAME_SKIP)
    test_env = get_player(
        args.rom,
        image_size=IMAGE_SIZE,
        frame_skip=FRAME_SKIP,
        context_len=CONTEXT_LEN)

    # Init Prioritized Replay Memory
    per = ProportionalPER(alpha=0.6, seg_num=args.batch_size, size=MEMORY_SIZE)

    # Prepare PARL agent
    act_dim = env.action_space.n
    model = AtariModel(act_dim)
    if args.alg == 'ddqn':
        algorithm = PrioritizedDoubleDQN(
            model, act_dim=act_dim, gamma=GAMMA, lr=LEARNING_RATE)
    elif args.alg == 'dqn':
        algorithm = PrioritizedDQN(
            model, act_dim=act_dim, gamma=GAMMA, lr=LEARNING_RATE)
    agent = AtariAgent(algorithm, act_dim=act_dim, update_freq=UPDATE_FREQ)

    # Replay memory warmup
    total_step = 0
    with tqdm(total=MEMORY_SIZE, desc='[Replay Memory Warm Up]') as pbar:
        mem = []
        while total_step < MEMORY_WARMUP_SIZE:
            total_reward, steps, _ = run_episode(
                env, agent, per, mem=mem, warmup=True)
            total_step += steps
            pbar.update(steps)
    per.elements.from_list(mem[:int(MEMORY_WARMUP_SIZE)])

    env_name = args.rom.split('/')[-1].split('.')[0]

    test_flag = 0
    total_steps = 0
    pbar = tqdm(total=args.train_total_steps)
    while total_steps < args.train_total_steps:
        # start epoch
        total_reward, steps, loss = run_episode(env, agent, per, train=True)
        total_steps += steps
        pbar.set_description('[train]exploration:{}'.format(agent.exploration))
        summary.add_scalar('{}/score'.format(env_name), total_reward,
                           total_steps)
        summary.add_scalar('{}/loss'.format(env_name), loss,
                           total_steps)  # mean of total loss
        summary.add_scalar('{}/exploration'.format(env_name),
                           agent.exploration, total_steps)
        pbar.update(steps)

        if total_steps // args.test_every_steps >= test_flag:
            while total_steps // args.test_every_steps >= test_flag:
                test_flag += 1
            pbar.write("testing")
            test_rewards = []
            for _ in tqdm(range(3), desc='eval agent'):
                eval_reward = run_evaluate_episode(test_env, agent)
                test_rewards.append(eval_reward)
            eval_reward = np.mean(test_rewards)
            logger.info(
                "eval_agent done, (steps, eval_reward): ({}, {})".format(
                    total_steps, eval_reward))
            summary.add_scalar('{}/eval'.format(env_name), eval_reward,
                               total_steps)

    pbar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--rom', help='path of the rom of the atari game', required=True)
    parser.add_argument(
        '--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument(
        '--alg',
        type=str,
        default="ddqn",
        help='dqn or ddqn, training algorithm to use.')
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
    args = parser.parse_args()
    assert args.alg in ['dqn','ddqn'], \
        'used algorithm should be dqn or ddqn (double dqn)'
    get_beta = beta_adder(init_beta=0.5)
    main()
