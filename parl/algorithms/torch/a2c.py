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
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import parl
from parl.utils.utils import check_model_method

__all__ = ['A2C']


class A2C(parl.Algorithm):
    def __init__(self, model, config):
        # checks
        assert isinstance(config['vf_loss_coeff'], (int, float))
        check_model_method(model, 'value', self.__class__.__name__)
        check_model_method(model, 'policy', self.__class__.__name__)
        check_model_method(model, 'policy_and_value', self.__class__.__name__)

        self.model = model
        self.vf_loss_coeff = config['vf_loss_coeff']
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=config['learning_rate'])
        self.config = config

    def learn(self, obs, actions, advantages, target_values, lr,
              entropy_coeff):
        logits = self.model.policy(obs)
        act_dim = logits.shape[-1]
        actions_onehot = F.one_hot(actions, act_dim)
        actions_log_probs = torch.sum(
            F.log_softmax(logits, dim=1) * actions_onehot, dim=-1)
        # The policy gradient loss
        pi_loss = -1.0 * torch.sum(actions_log_probs * advantages)

        # The value function loss
        values = self.model.value(obs)
        delta = values - target_values
        vf_loss = 0.5 * torch.sum(torch.square(delta))

        policy_distri = Categorical(logits=logits)
        # The entropy loss (We want to maximize entropy, so entropy_ceoff < 0)
        policy_entropy = policy_distri.entropy()
        entropy = torch.sum(policy_entropy)

        total_loss = pi_loss + vf_loss * self.vf_loss_coeff + entropy * entropy_coeff

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        total_loss.backward()
        # clip the grad
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=40.0)
        self.optimizer.step()
        self.optimizer.zero_grad()

        return total_loss, pi_loss, vf_loss, entropy

    def sample(self, obs):
        logits, values = self.model.policy_and_value(obs)
        sample_actions = Categorical(logits=logits).sample().long()
        return sample_actions, values

    def prob_and_value(self, obs):
        logits, values = self.model.policy_and_value(obs)
        probs = F.softmax(logits, dim=1)
        return probs, values

    def predict(self, obs):
        prob = self.model.policy(obs)
        _, predict_actions = prob.max(-1)
        return predict_actions

    def value(self, obs):
        values = self.model.value(obs)
        return values
