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
import torch.optim as optim
import parl
from torch.distributions import Categorical

__all__ = ['PolicyGradient']


class PolicyGradient(parl.Algorithm):
    def __init__(self, model, lr):
        """Policy gradient algorithm

        Args:
            model (parl.Model): model defining forward network of policy.
            lr (float): learning rate.

        """
        assert isinstance(lr, float)

        self.model = model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def predict(self, obs):
        """Predict the probability of actions

        Args:
            obs (torch.tensor): shape of (obs_dim,)

        Returns:
            prob (torch.tensor): shape of (action_dim,)
        """
        prob = self.model(obs)
        return prob

    def learn(self, obs, action, reward):
        """Update model with policy gradient algorithm

        Args:
            obs (torch.tensor): shape of (batch_size, obs_dim)
            action (torch.tensor): shape of (batch_size, 1)
            reward (torch.tensor): shape of (batch_size, 1)

        Returns:
            loss (torch.tensor): shape of (1)

        """
        prob = self.model(obs)

        log_prob = Categorical(prob).log_prob(action)

        loss = torch.mean(-1 * log_prob * reward)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
