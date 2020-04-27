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

import torch
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from random import random, randint

import parl
from parl.utils.scheduler import PiecewiseScheduler, LinearDecayScheduler

__all__ = ['A2C']


class A2C(parl.Algorithm):
    def __init__(self, model, config):
        assert isinstance(config['vf_loss_coeff'], (int, float))
        self.model = model
        self.vf_loss_coeff = config['vf_loss_coeff']
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=config['learning_rate'])
        self.config = config

        self.lr_scheduler = LinearDecayScheduler(config['start_lr'],
                                                 config['max_sample_steps'])

        self.entropy_coeff_scheduler = PiecewiseScheduler(
            config['entropy_coeff_scheduler'])

    def learn(self, obs, actions, advantages, target_values):
        prob = self.model.policy(obs, softmax_dim=1)
        policy_distri = Categorical(prob)
        actions_log_probs = policy_distri.log_prob(actions)

        # The policy gradient loss
        pi_loss = -((actions_log_probs * advantages).sum())

        # The value function loss
        values = self.model.value(obs).reshape(-1)
        delta = values - target_values
        vf_loss = 0.5 * torch.mul(delta, delta).sum()

        # The entropy loss (We want to maximize entropy, so entropy_ceoff < 0)
        policy_entropy = policy_distri.entropy()
        entropy = policy_entropy.sum()

        lr = self.lr_scheduler.step(step_num=obs.shape[0])
        entropy_coeff = self.entropy_coeff_scheduler.step()

        total_loss = pi_loss + vf_loss * self.vf_loss_coeff + entropy * entropy_coeff

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        total_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return total_loss, pi_loss, vf_loss, entropy, lr, entropy_coeff

    def sample(self, obs):
        prob, values = self.model.policy_and_value(obs)
        sample_actions = Categorical(prob).sample()

        return sample_actions, values

    def predict(self, obs):
        prob = self.model.policy(obs)
        _, predict_actions = prob.max(-1)

        return predict_actions

    def value(self, obs):
        values = self.model.value(obs)
        return values
