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

# modified from https://github.com/marlbenchmark/on-policy

import os
import time
import numpy as np
import torch
import argparse
from itertools import chain
from mpe.env_wrappers import ParallelEnv
from mappo_model import MAPPOModel
from parl.algorithms import MAPPO
from mappo_agent import MAPPOgent
from mappo_buffer import SeparatedReplayBuffer
from parl.utils import logger, summary

LR = 7e-4  # learning rate
VALUE_LOSS_COEF = 1  # Value loss coefficient (ie. c_1 in the paper)
ENTROPY_COEF = 0.01  # Entropy coefficient (ie. c_2 in the paper)
HUBER_DELTA = 10.0  # coefficience of huber loss
EPS = 1e-5  # Adam optimizer epsilon (default: 1e-5)
MAX_GRAD_NORM = 10.0  # Max gradient norm for gradient clipping
EPISODE_LENGTH = 25  # Max length for any episode
GAMMA = 0.99  # discount factor for rewards (default: 0.99)
GAE_LAMBDA = 0.95  # gae lambda parameter (default: 0.95)
LOG_INTERVAL_EPISODES = 5  # time duration between contiunous twice log printing


def main():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    envs = ParallelEnv(args.scenario_name, args.env_num, args.seed)

    agent_num = len(envs.observation_space)
    env_num = args.env_num

    agents = []
    buffers = []
    for agent_id in range(agent_num):
        share_observation_space = envs.share_observation_space[agent_id] if args.use_centralized_V else \
            envs.observation_space[agent_id]

        model = MAPPOModel(
            envs.observation_space[agent_id],
            share_observation_space,
            envs.action_space[agent_id],
            device=device)
        algorithm = MAPPO(
            model,
            args.clip_param,
            VALUE_LOSS_COEF,
            ENTROPY_COEF,
            LR,
            HUBER_DELTA,
            EPS,
            MAX_GRAD_NORM,
            args.use_popart,
            args.use_value_active_masks,
            device=device)
        agent = MAPPOgent(args.ppo_epoch, args.num_mini_batch, env_num,
                          algorithm, args.use_popart)
        # buffer
        bu = SeparatedReplayBuffer(
            EPISODE_LENGTH, env_num, GAMMA, GAE_LAMBDA,
            envs.observation_space[agent_id], share_observation_space,
            envs.action_space[agent_id], args.use_popart)
        agents.append(agent)
        buffers.append(bu)

    if args.restore:
        # restore modle
        for i in range(len(agents)):
            model_file = args.model_dir + '/agent_' + str(i)
            if not os.path.exists(model_file):
                raise Exception(
                    'model file {} does not exits'.format(model_file))
            agents[i].restore(model_file)

    start = time.time()
    obs = envs.reset()
    share_obs = []
    for o in obs:
        share_obs.append(list(chain(*o)))
    share_obs = np.array(share_obs)
    for agent_id in range(agent_num):
        if not args.use_centralized_V:
            share_obs = np.array(list(obs[:, agent_id]))
        buffers[agent_id].share_obs[0] = share_obs.copy()
        buffers[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()

    episodes = int(args.num_env_steps) // EPISODE_LENGTH // args.env_num
    for episode in range(episodes):
        for step in range(EPISODE_LENGTH):
            # Sample actions
            values = []
            actions = []
            action_log_probs = []
            for agent_id in range(agent_num):
                value, action, action_log_prob = agents[agent_id].sample(
                    buffers[agent_id].share_obs[step],
                    buffers[agent_id].obs[step])
                values.append(value)
                actions.append(action)
                action_log_probs.append(action_log_prob)

            obs, rewards, dones, infos = envs.step(actions)

            masks = np.ones((args.env_num, agent_num, 1), dtype=np.float32)
            masks[dones == True] = np.zeros(((dones == True).sum(), 1),
                                            dtype=np.float32)

            share_obs = []
            for o in obs:
                share_obs.append(list(chain(*o)))
            share_obs = np.array(share_obs)
            for agent_id in range(agent_num):
                if not args.use_centralized_V:
                    share_obs = np.array(list(obs[:, agent_id]))

                buffers[agent_id].insert(
                    share_obs, np.array(list(obs[:, agent_id])),
                    actions[agent_id], action_log_probs[agent_id],
                    values[agent_id], rewards[:, agent_id], masks[:, agent_id])

        # compute return and update network
        with torch.no_grad():
            for agent_id in range(agent_num):
                next_values = agents[agent_id].value(
                    buffers[agent_id].share_obs[-1]).squeeze(-1)
                buffers[agent_id].compute_returns(
                    next_values, agents[agent_id].value_normalizer)

        # learn
        train_infos = []
        for agent_id in range(agent_num):
            train_info = agents[agent_id].learn(buffers[agent_id])
            train_infos.append(train_info)
            buffers[agent_id].after_update()

        # log information
        total_num_steps = (episode + 1) * EPISODE_LENGTH * args.env_num
        if episode % LOG_INTERVAL_EPISODES == 0:
            end = time.time()
            agent_rewards = []
            for agent_id in range(agent_num):
                idv_rews = []
                for info in infos:
                    if 'individual_reward' in info[agent_id].keys():
                        idv_rews.append(info[agent_id]['individual_reward'])
                individual_rewards = round(np.mean(idv_rews), 3)
                average_episode_rewards = round(
                    np.mean(buffers[agent_id].rewards) * EPISODE_LENGTH, 3)
                agent_rewards.append(individual_rewards)
                train_infos[agent_id].update({
                    'individual_rewards':
                    individual_rewards
                })
                train_infos[agent_id].update({
                    "average_episode_rewards":
                    average_episode_rewards
                })

            use_time = round(end - start, 3)
            logger.info(
                'Steps: {}, Episodes: {}/{}, Mean episode reward: {}, mean agents rewards {}, Time: {}'
                .format(total_num_steps, episode, episodes,
                        average_episode_rewards, agent_rewards, use_time))

            for agent_id in range(agent_num):
                for k, v in train_infos[agent_id].items():
                    agent_k = "agent%i/" % agent_id + k
                    summary.add_scalar(agent_k, v, total_num_steps)

            # save model
            if not args.restore:
                model_dir = args.model_dir
                os.makedirs(os.path.dirname(model_dir), exist_ok=True)
                for i in range(len(agents)):
                    model_name = '/agent_' + str(i)
                    agents[i].save(model_dir + model_name)

    envs.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--scenario_name',
        type=str,
        default='simple_speaker_listener',
        choices=['simple_spread', 'simple_reference'],
        help='scenario of MultiAgentEnv')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument(
        '--env_num',
        type=int,
        default=128,
        help='Number of parallel envs on each python process')
    parser.add_argument(
        '--num_env_steps',
        type=int,
        default=2e7,
        help='Number of environment steps to train')
    parser.add_argument(
        '--restore',
        action='store_true',
        default=False,
        help='restore or not, must have model_dir')
    parser.add_argument(
        '--model_dir',
        type=str,
        default='./model',
        help='directory for saving model')

    # Five suggestions mentioned in the paper
    parser.add_argument(
        '--use_popart',
        default=True,
        help=
        'whether to use PopArt to normalize rewards, suggestion 1 in the paper (default: True)'
    )
    parser.add_argument(
        '--use_centralized_V',
        default=True,
        help=
        'whether to use centralized V function, suggestion 2 in the paper (default: True)'
    )
    parser.add_argument(
        "--ppo_epoch",
        type=int,
        default=15,
        help=
        'number of epochs for updating using each T data, suggestion 3 in the paper (default: 15)'
    )
    parser.add_argument(
        "--num_mini_batch",
        type=int,
        default=1,
        help='number of batches for ppo, suggestion 3 in the paper (default: 1)'
    )
    parser.add_argument(
        "--clip_param",
        type=float,
        default=0.2,
        help='ppo clip parameter, suggestion 4 in the paper (default: 0.2)')
    parser.add_argument(
        "--use_value_active_masks",
        default=True,
        help=
        "whether to mask useless data in value loss, suggestion 5 in the paper (default: True)"
    )

    args = parser.parse_args()
    logger.set_dir('./train_log/' + str(args.scenario_name))

    main()
