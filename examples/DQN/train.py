#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
import numpy as np
import os
from atari import AtariPlayer
from atari_agent import AtariAgent
from atari_model import AtariModel
from atari_wrapper import FrameStack, MapState, FireResetEnv, LimitLength
from collections import deque
from datetime import datetime
from expreplay import ReplayMemory, Experience
from parl.algorithms import DQN
from parl.utils import logger
from tqdm import tqdm

MEMORY_SIZE = 1e6
MEMORY_WARMUP_SIZE = MEMORY_SIZE // 20
IMAGE_SIZE = (84, 84)
CONTEXT_LEN = 4
ACTION_REPEAT = 4  # aka FRAME_SKIP
UPDATE_FREQ = 4
GAMMA = 0.99
LEARNING_RATE = 1e-3 * 0.5


def run_train_episode(agent, env, exp):
    total_reward = 0
    all_cost = []
    state = env.reset()
    step = 0
    while True:
        step += 1
        context = exp.recent_state()
        context.append(state)
        context = np.stack(context, axis=0)
        action = agent.sample(context)
        next_state, reward, isOver, _ = env.step(action)
        exp.append(Experience(state, action, reward, isOver))
        # start training
        if len(exp) > MEMORY_WARMUP_SIZE:
            if step % UPDATE_FREQ == 0:
                batch_all_state, batch_action, batch_reward, batch_isOver = exp.sample_batch(
                    args.batch_size)
                batch_state = batch_all_state[:, :CONTEXT_LEN, :, :]
                batch_next_state = batch_all_state[:, 1:, :, :]
                cost = agent.learn(batch_state, batch_action, batch_reward,
                                   batch_next_state, batch_isOver)
                all_cost.append(float(cost))
        total_reward += reward
        state = next_state
        if isOver:
            break
    logger.info('[Train]total_reward: {}, mean_cost: {}'.format(
        total_reward, np.mean(all_cost)))
    return total_reward, step


def get_player(rom, viz=False, train=False):
    env = AtariPlayer(
        rom,
        frame_skip=ACTION_REPEAT,
        viz=viz,
        live_lost_as_eoe=train,
        max_num_frames=60000)
    env = FireResetEnv(env)
    env = MapState(env, lambda im: cv2.resize(im, IMAGE_SIZE))
    if not train:
        # in training, context is taken care of in expreplay buffer
        env = FrameStack(env, CONTEXT_LEN)
    return env


def eval_agent(agent, env):
    episode_reward = []
    for _ in tqdm(range(30), desc='eval agent'):
        state = env.reset()
        total_reward = 0
        step = 0
        while True:
            step += 1
            action = agent.predict(state)
            state, reward, isOver, info = env.step(action)
            total_reward += reward
            if isOver:
                break
        episode_reward.append(total_reward)
    eval_reward = np.mean(episode_reward)
    return eval_reward


def train_agent():
    env = get_player(args.rom, train=True)
    test_env = get_player(args.rom)
    exp = ReplayMemory(MEMORY_SIZE, IMAGE_SIZE, CONTEXT_LEN)
    action_dim = env.action_space.n

    hyperparas = {
        'action_dim': action_dim,
        'lr': LEARNING_RATE,
        'gamma': GAMMA
    }
    model = AtariModel(IMAGE_SIZE[0], IMAGE_SIZE[1], action_dim)
    algorithm = DQN(model, hyperparas)
    agent = AtariAgent(algorithm, action_dim)

    with tqdm(total=MEMORY_WARMUP_SIZE) as pbar:
        while len(exp) < MEMORY_WARMUP_SIZE:
            total_reward, step = run_train_episode(agent, env, exp)
            pbar.update(step)

    # train
    test_flag = 0
    pbar = tqdm(total=1e8)
    recent_100_reward = []
    total_step = 0
    max_reward = None
    while True:
        # start epoch
        total_reward, step = run_train_episode(agent, env, exp)
        total_step += step
        pbar.set_description('[train]exploration:{}'.format(agent.exploration))
        pbar.update(step)

        if total_step // args.test_every_steps == test_flag:
            pbar.write("testing")
            eval_reward = eval_agent(agent, test_env)
            test_flag += 1
            logger.info(
                "eval_agent done, (steps, eval_reward): ({}, {})".format(
                    total_step, eval_reward))

    pbar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rom', help='atari rom', required=True)
    parser.add_argument(
        '--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument(
        '--test_every_steps',
        type=int,
        default=100000,
        help='every steps number to run test')
    args = parser.parse_args()
    train_agent()
