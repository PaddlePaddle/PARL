#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import torch

import argparse
import pickle
import random
import sys

from tqdm import tqdm
import parl
from model import TrajectoryModel
from agent import DTAgent
from parl.utils import summary
from evaluate_episodes import eval_episodes
from parl.utils import logger


def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env_name, dataset = config['env'], config['dataset']

    if env_name == 'hopper':
        env = gym.make('Hopper-v3')
        env_targets = 3600
    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v3')
        env_targets = 12000
    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v3')
        env_targets = 5000
    else:
        raise NotImplementedError

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    logger.info('Starting new experiment: {} {}'.format(env_name, dataset))

    K = config['K']
    batch_size = config['batch_size']
    num_eval_episodes = config['num_eval_episodes']
    pct_traj = config.get('pct_traj', 1.)

    model = TrajectoryModel(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=K,
        max_ep_len=config['max_ep_len'],
        hidden_size=config['embed_dim'],
        n_layer=config['n_layer'],
        n_head=config['n_head'],
        n_inner=4 * config['embed_dim'],
        activation_function=config['activation_function'],
        n_positions=1024,
        resid_pdrop=config['dropout'],
        attn_pdrop=config['dropout'],
    )

    model = model.to(device=device)
    alg = parl.algorithms.DecisionTransformer(model, config['learning_rate'],
                                              config['warmup_steps'],
                                              config['weight_decay'])
    agent = DTAgent(alg, config)

    dataset_path = f'data/{env_name}-{dataset}-v2.pkl'
    state_mean, state_std = agent.load_data(dataset_path)

    for iter in range(config['max_iters']):
        train_loss = []
        agent.train()
        for step in tqdm(range(config['num_steps_per_iter'])):
            loss = agent.learn()
            train_loss.append(loss)
        logger.info("[training] iter:{} loss:{}".format(
            iter, np.mean(train_loss)))
        summary.add_scalar('train_loss', np.mean(train_loss), iter)

        agent.eval()
        logs = eval_episodes(env_targets, env, state_dim, act_dim, agent,
                             config['max_ep_len'], config['rew_scale'],
                             state_mean, state_std, device)
        for k, v in logs.items():
            logger.info('{}: {}'.format(k, v))
            summary.add_scalar(k, v, iter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='hopper')
    parser.add_argument(
        '--dataset', type=str,
        default='medium')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument(
        '--rew_scale', type=float,
        default=1000)  #normalization for rewards/returns
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    parser.add_argument('--max_ep_len', type=int, default=1000)

    args = parser.parse_args()
    logger.set_dir(f'./train_log/{args.env}_{args.dataset}')

    main(config=vars(args))
