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
from parl.utils import tensorboard, logger

from agent import AtariAgent
from algorithm import DQN
from atari_wrapper import FireResetEnv, FrameStack, LimitLength, MapState
from evaluate import EvalModel
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
    state = env.reset()
    steps = 0
    while True:
        steps += 1
        context = rpm.recent_state()
        context.append(state)
        context = np.stack(context, axis=0)
        action = agent.sample(context)
        next_state, reward, isOver, _ = env.step(action)
        rpm.append(Experience(state, action, reward, isOver))
        if rpm.size() > MEMORY_WARMUP_SIZE:
            if steps % UPDATE_FREQ == 0:
                batch_all_state, batch_action, batch_reward, batch_isOver = rpm.sample_batch(
                    args.batch_size)
                batch_state = batch_all_state[:, :CONTEXT_LEN, :, :]
                batch_next_state = batch_all_state[:, 1:, :, :]
                cost = agent.learn(batch_state, batch_action, batch_reward,
                                   batch_next_state, batch_isOver)
                all_cost.append(cost)
        total_reward += reward
        state = next_state
        if isOver:
            mean_loss = np.mean(all_cost) if all_cost else None
            return total_reward, steps, mean_loss


def get_fixed_states(rpm, batch_size):
    states = []
    for _ in range(3):
        batch_all_state = rpm.sample_batch(batch_size)[0]
        batch_state = batch_all_state[:, :CONTEXT_LEN, :, :]
        states.append(batch_state)
    fixed_states = np.concatenate(states, axis=0)
    return fixed_states


def evaluate_fixed_Q(agent, states):
    with torch.no_grad():
        max_pred_Q = agent.alg.model(states).max(1)[0].mean()
    return max_pred_Q.item()


def get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item()**2
    total_norm = total_norm**(1. / 2)
    return total_norm


def get_evaluator(args, act_dim):
    config = {
        'rom': args.rom,
        'image_size': IMAGE_SIZE,
        'frame_skip': FRAME_SKIP,
        'context_len': CONTEXT_LEN,
        'gamma': GAMMA,
        'act_dim': act_dim,
        'algo': args.algo,
        'lr': args.lr,
        'eval_nums': args.eval_nums,
        'actor_nums': args.actor_nums
    }

    evaluator = EvalModel(config)
    return evaluator


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
    algorithm = DQN(
        model, act_dim=act_dim, gamma=GAMMA, lr=args.lr, algo=args.algo)
    agent = AtariAgent(algorithm, act_dim=act_dim)

    with tqdm(
            total=MEMORY_WARMUP_SIZE,
            desc='[Replay Memory Warm Up]') as pbar:
        while rpm.size() < MEMORY_WARMUP_SIZE:
            total_reward, steps, _ = run_train_episode(env, agent, rpm)
            pbar.update(steps)

    # Get fixed states to check value function.
    fixed_states = get_fixed_states(rpm, args.batch_size)
    fixed_states = torch.tensor(fixed_states, dtype=torch.float, device=device)

    # train
    test_flag = 0
    total_steps = 0

    # run ``xparl start --port 1234`` to start a parl cluster
    parl.connect('localhost:1234', distributed_files=[args.rom])
    evaluator = get_evaluator(args, act_dim)
    th = threading.Thread(target=evaluator.run)
    th.start()

    with tqdm(total=args.train_total_steps, desc='[Training Model]') as pbar:
        while total_steps < args.train_total_steps:
            total_reward, steps, loss = run_train_episode(env, agent, rpm)
            total_steps += steps
            tensorboard.add_scalar('dqn/score', total_reward, total_steps)
            tensorboard.add_scalar('dqn/loss', loss, total_steps)
            tensorboard.add_scalar('dqn/exploration', agent.exploration,
                                   total_steps)
            tensorboard.add_scalar('dqn/grad_norm',
                                   get_grad_norm(agent.alg.model), total_steps)
            pbar.update(steps)

            if total_steps // args.test_every_steps >= test_flag:
                while total_steps // args.test_every_steps >= test_flag:
                    test_flag += 1
                eval_rewards = []
                latest_weights = [weight.detach().cpu().numpy() for weight in agent.alg.get_weights()]
                evaluator.weights_queue.put([latest_weights, total_steps])
                tensorboard.add_scalar('dqn/Q value',
                                       evaluate_fixed_Q(agent, fixed_states),
                                       total_steps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rom', default='rom_files/breakout.bin')
    parser.add_argument(
        '--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--lr', default=3e-4, help='learning_rate')
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
    parser.add_argument(
        '--algo', type=str, default='Dueling', help='Which DQN model to use.')
    parser.add_argument('--eval_nums', default=5)
    parser.add_argument('--actor_nums', default=2)
    args = parser.parse_args()
    logger.set_dir(os.path.join('./train_log', str(args.algo)))
    main()
