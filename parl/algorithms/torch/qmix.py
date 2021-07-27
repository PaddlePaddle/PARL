#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import torch.nn.functional as F
import parl
from parl.utils.utils import check_model_method
import numpy as np
from copy import deepcopy

__all__ = ['QMIX']


class QMIX(parl.Algorithm):
    def __init__(self,
                 agent_model,
                 qmixer_model,
                 double_q=True,
                 gamma=0.99,
                 lr=0.0005,
                 clip_grad_norm=None):
        """ QMIX algorithm
        Args:
            agent_model (parl.Model): agents' local q network for decision making.
            qmixer_model (parl.Model): A mixing network which takes local q values as input
                to construct a global Q network.
            double_q (bool): Double-DQN.
            gamma (float): discounted factor for reward computation.
            lr (float): learning rate.
            clip_grad_norm (None, or float): clipped value of gradients' global norm.
        """
        # checks
        check_model_method(agent_model, 'init_hidden', self.__class__.__name__)
        check_model_method(agent_model, 'forward', self.__class__.__name__)
        check_model_method(qmixer_model, 'forward', self.__class__.__name__)
        assert hasattr(qmixer_model, 'n_agents') and not callable(
            getattr(qmixer_model, 'n_agents',
                    None)), 'qmixer_model needs to have attribute n_agents'
        assert isinstance(gamma, float)
        assert isinstance(lr, float)

        self.agent_model = agent_model
        self.qmixer_model = qmixer_model
        self.target_agent_model = deepcopy(self.agent_model)
        self.target_qmixer_model = deepcopy(self.qmixer_model)

        self.n_agents = self.qmixer_model.n_agents

        self.double_q = double_q
        self.gamma = gamma
        self.lr = lr
        self.clip_grad_norm = clip_grad_norm

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.agent_model.to(device)
        self.target_agent_model.to(device)
        self.qmixer_model.to(device)
        self.target_qmixer_model.to(device)

        self.params = list(self.agent_model.parameters())
        self.params += self.qmixer_model.parameters()
        self.optimizer = torch.optim.RMSprop(
            params=self.params, lr=self.lr, alpha=0.99, eps=0.00001)

    def _init_hidden_states(self, batch_size):
        self.hidden_states = self.agent_model.init_hidden().unsqueeze(
            0).expand(batch_size, self.n_agents, -1)
        self.target_hidden_states = self.target_agent_model.init_hidden(
        ).unsqueeze(0).expand(batch_size, self.n_agents, -1)

    def predict_local_q(self, obs, hidden_state):
        '''
        Args:
            obs (torch.Tensor): (n_agents, obs_shape)
        Returns:
            self.agent_model(obs, hidden_state)
        '''
        return self.agent_model(obs, hidden_state)

    def learn(self, state_batch, actions_batch, reward_batch, terminated_batch,
              obs_batch, available_actions_batch, filled_batch):
        '''
        Args:
            state_batch (torch.Tensor):             (batch_size, T, state_shape)
            actions_batch (torch.Tensor):           (batch_size, T, n_agents)
            reward_batch (torch.Tensor):            (batch_size, T, 1)
            terminated_batch (torch.Tensor):        (batch_size, T, 1)
            obs_batch (torch.Tensor):               (batch_size, T, n_agents, obs_shape)
            available_actions_batch (torch.Tensor): (batch_size, T, n_agents, n_actions)
            filled_batch (torch.Tensor):            (batch_size, T, 1)
        Returns:
            loss (float): train loss
            td_error (float): train TD error
        '''
        batch_size = state_batch.shape[0]
        episode_len = state_batch.shape[1]
        self._init_hidden_states(batch_size)

        reward_batch = reward_batch[:, :-1, :]
        actions_batch = actions_batch[:, :-1, :].unsqueeze(-1)
        terminated_batch = terminated_batch[:, :-1, :]
        filled_batch = filled_batch[:, :-1, :]

        mask = (1 - filled_batch) * (1 - terminated_batch)

        local_qs = []
        target_local_qs = []
        for t in range(episode_len):
            obs = obs_batch[:, t, :, :]
            obs = obs.reshape(-1, obs_batch.shape[-1])
            local_q, self.hidden_states = self.agent_model(
                obs, self.hidden_states)
            local_q = local_q.reshape(batch_size, self.n_agents, -1)
            local_qs.append(local_q)

            target_local_q, self.target_hidden_states = self.target_agent_model(
                obs, self.target_hidden_states)
            target_local_q = target_local_q.view(batch_size, self.n_agents, -1)
            target_local_qs.append(target_local_q)

        local_qs = torch.stack(local_qs, dim=1)
        target_local_qs = torch.stack(target_local_qs[1:], dim=1)

        chosen_action_local_qs = torch.gather(
            local_qs[:, :-1, :, :], dim=3, index=actions_batch).squeeze(3)
        # mask unavailable actions
        target_local_qs[available_actions_batch[:, 1:, :] == 0] = -1e10
        if self.double_q:
            local_qs_detach = local_qs.clone().detach()
            local_qs_detach[available_actions_batch == 0] = -1e10
            cur_max_actions = local_qs_detach[:, 1:].max(
                dim=3, keepdim=True)[1]
            target_local_max_qs = torch.gather(target_local_qs, 3,
                                               cur_max_actions).squeeze(3)
        else:
            target_local_max_qs = target_local_qs.max(
                dim=3)[0]  # idx0: value, idx1: index

        chosen_action_global_qs = self.qmixer_model(chosen_action_local_qs,
                                                    state_batch[:, :-1, :])
        target_global_max_qs = self.target_qmixer_model(
            target_local_max_qs, state_batch[:, 1:, :])

        target = reward_batch + self.gamma * (
            1 - terminated_batch) * target_global_max_qs
        td_error = target.detach() - chosen_action_global_qs
        masked_td_error = td_error * mask
        mean_td_error = masked_td_error.sum() / mask.sum()
        loss = (masked_td_error**2).sum() / mask.sum()

        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.params, self.clip_grad_norm)
        self.optimizer.step()
        return loss.item(), mean_td_error.item()
