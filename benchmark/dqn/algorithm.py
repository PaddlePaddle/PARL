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

import warnings
warnings.simplefilter('default')

import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from parl.core.torch.algorithm import Algorithm
from parl.utils.deprecation import deprecated
import numpy as np

__all__ = ['DQN']


class DQN(Algorithm):
    def __init__(self, model, algo='DQN', act_dim=None, gamma=None, lr=None):
        """ DQN algorithm
        
        Args:
            model (parl.Model): model defining forward network of Q function.
            algo (str): which dqn model to use.
            hyperparas (dict): (deprecated) dict of hyper parameters.
            act_dim (int): dimension of the action space
            gamma (float): discounted factor for reward computation.
            lr (float): learning rate.
        """
        self.model = model
        self.target_model = copy.deepcopy(model)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.target_model.to(device)

        assert isinstance(act_dim, int)
        assert isinstance(gamma, float)
        assert isinstance(lr, float)
        self.act_dim = act_dim
        self.gamma = gamma
        self.lr = lr
        self.algo = algo

        self.mse_loss = torch.nn.MSELoss()
        self.lr_lambda = lambda epoch: (1 - epoch * 4e-7)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer,
                                                     self.lr_lambda)

    def predict(self, obs):
        """ use value model self.model to predict the action value
        """
        with torch.no_grad():
            pred_q = self.model(obs)
        return pred_q

    def learn(self, obs, action, reward, next_obs, terminal):
        """ update value model self.model with DQN algorithm
        """
        pred_value = self.model(obs).gather(1, action)
        greedy_action = self.model(next_obs).max(dim=1, keepdim=True)[1]
        with torch.no_grad():
            if self.algo == 'Double':
                max_v = self.target_model(next_obs).gather(1, greedy_action)
            else:
                max_v = self.target_model(next_obs).max(1, keepdim=True)[0]
            target = reward + (1 - terminal) * self.gamma * max_v
        self.optimizer.zero_grad()
        loss = self.mse_loss(pred_value, target)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    def sync_target(self):
        self.model.sync_weights_to(self.target_model)
