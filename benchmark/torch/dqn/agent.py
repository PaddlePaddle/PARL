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
import gym

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import parl


class AtariAgent(parl.Agent):
    """Base class of the Agent.

    Args:
        algorithm (object): Algorithm used by this agent.
        args (argparse.Namespace): Model configurations.
        device (torch.device): use cpu or gpu.
    """

    def __init__(self, algorithm, act_dim):
        assert isinstance(act_dim, int)
        super(AtariAgent, self).__init__(algorithm)
        self.act_dim = act_dim
        self.exploration = 1
        self.global_step = 0
        self.update_target_steps = 10000 // 4

        self.device = torch.device('cuda' if torch.cuda.
                                   is_available() else 'cpu')

    def save(self, filepath):
        state = {
            'model': self.alg.model.state_dict(),
            'target_model': self.alg.target_model.state_dict(),
            'optimizer': self.alg.optimizer.state_dict(),
            'scheduler': self.alg.scheduler.state_dict(),
            'exploration': self.exploration,
        }
        torch.save(state, filepath)

    def restore(self, filepath):
        checkpoint = torch.load(filepath)
        self.exploration = checkpoint['exploration']
        self.alg.model.load_state_dict(checkpoint['model'])
        self.alg.target_model.load_state_dict(checkpoint['target_model'])
        self.alg.optimizer.load_state_dict(checkpoint['optimizer'])
        self.alg.scheduler.load_state_dict(checkpoint['scheduler'])

    def sample(self, obs):
        sample = np.random.random()
        if sample < self.exploration:
            act = np.random.randint(self.act_dim)
        else:
            if np.random.random() < 0.01:
                act = np.random.randint(self.act_dim)
            else:
                pred_q = self.predict(obs)
                act = pred_q.max(1)[1].item()
        self.exploration = max(0.1, self.exploration - 1e-6)
        return act

    def predict(self, obs):
        obs = np.expand_dims(obs, 0)
        obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        pred_q = self.alg.predict(obs)
        return pred_q

    def learn(self, obs, act, reward, next_obs, terminal):
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1

        act = np.expand_dims(act, -1)
        terminal = np.expand_dims(terminal, -1)
        reward = np.expand_dims(reward, -1)
        reward = np.clip(reward, -1, 1)

        obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        next_obs = torch.tensor(
            next_obs, dtype=torch.float, device=self.device)
        act = torch.tensor(act, dtype=torch.long, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float, device=self.device)
        terminal = torch.tensor(
            terminal, dtype=torch.float, device=self.device)

        cost = self.alg.learn(obs, act, reward, next_obs, terminal)
        return cost
