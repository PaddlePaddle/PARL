#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import cv2
import gym
import os
import threading
import torch
import parl

import numpy as np
from tqdm import tqdm
from parl.utils import summary, logger
from parl.algorithms import DQN, DDQN

from agent import AtariAgent
from atari_wrapper import FireResetEnv, FrameStack, LimitLength
from model import AtariModel
from replay_memory import ReplayMemory, Experience
from utils import get_player

MEMORY_SIZE = int(1e6)
MEMORY_WARMUP_SIZE = MEMORY_SIZE // 20
IMAGE_SIZE = (84, 84)
CONTEXT_LEN = 4
FRAME_SKIP = 4
UPDATE_FREQ = 4
GAMMA = 0.99


def run_train_episode(env, agent, rpm):
    total_reward = 0
    all_cost = []
    obs = env.reset()
    steps = 0
    while True:
        steps += 1
        context = rpm.recent_obs()
        context.append(obs)
        context = np.stack(context, axis=0)
        action = agent.sample(context)
        next_obs, reward, isOver, _ = env.step(action)
        rpm.append(Experience(obs, action, reward, isOver))
        if rpm.size() > MEMORY_WARMUP_SIZE:
            if steps % UPDATE_FREQ == 0:
                batch_all_obs, batch_action, batch_reward, batch_isOver = rpm.sample_batch(
                    args.batch_size)
                batch_obs = batch_all_obs[:, :CONTEXT_LEN, :, :]
                batch_next_obs = batch_all_obs[:, 1:, :, :]
                cost = agent.learn(batch_obs, batch_action, batch_reward,
                                   batch_next_obs, batch_isOver)
                all_cost.append(cost)
        total_reward += reward
        obs = next_obs
        if isOver:
            mean_loss = np.mean(all_cost) if all_cost else None
            return total_reward, steps, mean_loss


def run_evaluate_episode(env, agent):
    obs = env.reset()
    total_reward = 0
    while True:
        pred_Q = agent.predict(obs)
        action = pred_Q.max(1)[1].item()
        obs, reward, isOver, _ = env.step(action)
        total_reward += reward
        if isOver:
            return total_reward


def get_fixed_obs(rpm, batch_size):
    obs = []
    for _ in range(3):
        batch_all_obs = rpm.sample_batch(batch_size)[0]
        batch_obs = batch_all_obs[:, :CONTEXT_LEN, :, :]
        obs.append(batch_obs)
    fixed_obs = np.concatenate(obs, axis=0)
    return fixed_obs


def evaluate_fixed_Q(agent, obs):
    with torch.no_grad():
        max_pred_Q = agent.alg.model(obs).max(1)[0].mean()
    return max_pred_Q.item()


def get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item()**2
    total_norm = total_norm**(1. / 2)
    return total_norm


def main():
    env = get_player(
        args.rom, image_size=IMAGE_SIZE, train=True, frame_skip=FRAME_SKIP)
    test_env = get_player(
        args.rom,
        image_size=IMAGE_SIZE,
        frame_skip=FRAME_SKIP,
        context_len=CONTEXT_LEN)
    rpm = ReplayMemory(MEMORY_SIZE, IMAGE_SIZE, CONTEXT_LEN)
    act_dim = env.action_space.n
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AtariModel(CONTEXT_LEN, act_dim, args.algo)
    if args.algo in ['DQN', 'Dueling']:
        algorithm = DQN(model, gamma=GAMMA, lr=args.lr)
    elif args.algo is 'Double':
        algorithm = DDQN(model, gamma=GAMMA, lr=args.lr)
    agent = AtariAgent(algorithm, act_dim=act_dim)

    with tqdm(
            total=MEMORY_WARMUP_SIZE, desc='[Replay Memory Warm Up]') as pbar:
        while rpm.size() < MEMORY_WARMUP_SIZE:
            total_reward, steps, _ = run_train_episode(env, agent, rpm)
            pbar.update(steps)

    # Get fixed obs to check value function.
    fixed_obs = get_fixed_obs(rpm, args.batch_size)
    fixed_obs = torch.tensor(fixed_obs, dtype=torch.float, device=device)

    # train
    test_flag = 0
    total_steps = 0

    with tqdm(total=args.train_total_steps, desc='[Training Model]') as pbar:
        while total_steps < args.train_total_steps:
            total_reward, steps, loss = run_train_episode(env, agent, rpm)
            total_steps += steps
            pbar.update(steps)
            if total_steps // args.test_every_steps >= test_flag:
                while total_steps // args.test_every_steps >= test_flag:
                    test_flag += 1

                eval_rewards = []
                for _ in range(3):
                    eval_rewards.append(run_evaluate_episode(test_env, agent))

                summary.add_scalar('dqn/eval', np.mean(eval_rewards),
                                   total_steps)
                summary.add_scalar('dqn/score', total_reward, total_steps)
                summary.add_scalar('dqn/loss', loss, total_steps)
                summary.add_scalar('dqn/exploration', agent.exploration,
                                   total_steps)
                summary.add_scalar('dqn/Q value',
                                   evaluate_fixed_Q(agent, fixed_obs),
                                   total_steps)
                summary.add_scalar('dqn/grad_norm',
                                   get_grad_norm(agent.alg.model), total_steps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rom', default='rom_files/breakout.bin')
    parser.add_argument(
        '--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--lr', default=3e-4, help='learning_rate')
    parser.add_argument('--algo', default='DQN', help='DQN/Double/Dueling DQN')
    parser.add_argument(
        '--train_total_steps',
        type=int,
        default=int(1e7),
        help='maximum environmental steps of games')
    parser.add_argument(
        '--test_every_steps',
        type=int,
        default=int(1e5),
        help='the step interval between two consecutive evaluations')
    args = parser.parse_args()
    rom_name = args.rom.split('/')[-1].split('.')[0]
    logger.set_dir(os.path.join('./train_log', rom_name))
    main()
