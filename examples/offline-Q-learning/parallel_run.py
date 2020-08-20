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
import gym
import numpy as np
import os
import time
from tqdm import tqdm

import parl
import paddle.fluid as fluid
from parl.utils import get_gpu_count
from parl.utils import summary, logger

from dqn import DQN  # slight changes from parl.algorithms.DQN
from atari_agent import AtariAgent
from atari_model import AtariModel
from replay_memory import ReplayMemory, Experience
from utils import get_player

MEMORY_SIZE = int(1e6)
MEMORY_WARMUP_SIZE = MEMORY_SIZE // 20
IMAGE_SIZE = (84, 84)
CONTEXT_LEN = 4
FRAME_SKIP = 4
UPDATE_FREQ = 4
GAMMA = 0.99
LEARNING_RATE = 3e-4

gpu_num = get_gpu_count()


def run_train_step(agent, rpm):
    for step in range(args.train_total_steps):
        # use the first 80% data to train
        batch_all_obs, batch_action, batch_reward, batch_isOver = rpm.sample_batch(
            args.batch_size * gpu_num)
        batch_obs = batch_all_obs[:, :CONTEXT_LEN, :, :]
        batch_next_obs = batch_all_obs[:, 1:, :, :]
        cost = agent.learn(batch_obs, batch_action, batch_reward,
                           batch_next_obs, batch_isOver)

        if step % 100 == 0:
            # use the last 20% data to evaluate
            batch_all_obs, batch_action, batch_reward, batch_isOver = rpm.sample_test_batch(
                args.batch_size)
            batch_obs = batch_all_obs[:, :CONTEXT_LEN, :, :]
            batch_next_obs = batch_all_obs[:, 1:, :, :]
            eval_cost = agent.supervised_eval(batch_obs, batch_action,
                                              batch_reward, batch_next_obs,
                                              batch_isOver)
            logger.info(
                "train step {}, train costs are {}, eval cost is {}.".format(
                    step, cost, eval_cost))


def collect_exp(env, rpm, agent):
    obs = env.reset()
    # collect data to fulfill replay memory
    for i in tqdm(range(MEMORY_SIZE)):
        context = rpm.recent_obs()
        context.append(obs)
        context = np.stack(context, axis=0)
        action = agent.sample(context)

        next_obs, reward, isOver, _ = env.step(action)
        rpm.append(Experience(obs, action, reward, isOver))
        obs = next_obs


def main():
    env = get_player(
        args.rom, image_size=IMAGE_SIZE, train=True, frame_skip=FRAME_SKIP)
    file_path = "memory.npz"
    rpm = ReplayMemory(
        MEMORY_SIZE,
        IMAGE_SIZE,
        CONTEXT_LEN,
        load_file=True,  # load replay memory data from file
        file_path=file_path)
    act_dim = env.action_space.n

    model = AtariModel(act_dim)
    algorithm = DQN(
        model, act_dim=act_dim, gamma=GAMMA, lr=LEARNING_RATE * gpu_num)
    agent = AtariAgent(
        algorithm, act_dim=act_dim, total_step=args.train_total_steps)
    if os.path.isfile('./model.ckpt'):
        logger.info("load model from file")
        agent.restore('./model.ckpt')

    if args.train:
        logger.info("train with memory data")
        run_train_step(agent, rpm)
        logger.info("finish training. Save the model.")
        agent.save('./model.ckpt')
    else:
        logger.info("collect experience")
        collect_exp(env, rpm, agent)
        rpm.save_memory()
        logger.info("finish collecting, save successfully")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--rom', help='path of the rom of the atari game', required=True)
    parser.add_argument(
        '--batch_size', type=int, default=64, help='batch size for each GPU')
    parser.add_argument(
        '--train',
        action="store_true",
        help='update the value function (default: False)')
    parser.add_argument(
        '--train_total_steps',
        type=int,
        default=int(1e6),
        help='maximum environmental steps of games')

    args = parser.parse_args()
    main()
