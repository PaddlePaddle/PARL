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

# modified from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail

import copy
import os
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils
from arguments import get_args
from wrapper import make_env
from mujoco_model import MujocoModel
from parl.algorithms import PPO
from mujoco_agent import MujocoAgent
from storage import RolloutStorage
from evaluation import evaluate


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_env(args.env_name, args.seed, args.gamma)

    model = MujocoModel(envs.observation_space.shape[0],
                        envs.action_space.shape[0])
    model.to(device)

    algorithm = PPO(
        model,
        args.clip_param,
        args.value_loss_coef,
        args.entropy_coef,
        initial_lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm)

    agent = MujocoAgent(algorithm, device)

    rollouts = RolloutStorage(args.num_steps, envs.observation_space.shape[0],
                              envs.action_space.shape[0])

    obs = envs.reset()
    rollouts.obs[0] = np.copy(obs)

    episode_rewards = deque(maxlen=10)

    num_updates = int(args.num_env_steps) // args.num_steps
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(algorithm.optimizer, j, num_updates,
                                         args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob = agent.sample(
                    rollouts.obs[step])  # why use obs from rollouts???有病吧

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.append(obs, action, action_log_prob, value, reward, masks,
                            bad_masks)

        with torch.no_grad():
            next_value = agent.value(rollouts.obs[-1])

        value_loss, action_loss, dist_entropy = agent.learn(
            next_value, args.gamma, args.gae_lambda, args.ppo_epoch,
            args.num_mini_batch, rollouts)

        rollouts.after_update()

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_steps
            print(
                "Updates {}, num timesteps {},\n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps, len(episode_rewards),
                        np.mean(episode_rewards), np.median(episode_rewards),
                        np.min(episode_rewards), np.max(episode_rewards),
                        dist_entropy, value_loss, action_loss))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            eval_mean_reward = evaluate(agent, ob_rms, args.env_name,
                                        args.seed, device)


if __name__ == "__main__":
    main()
