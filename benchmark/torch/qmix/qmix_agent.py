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

import parl
import torch
import numpy as np
import os
import torch


class QMixAgent(parl.Agent):
    def __init__(self, algorithm, exploration_start, min_exploration,
                 exploration_decay, update_target_interval):
        self.alg = algorithm
        self.global_step = 0
        self.exploration = exploration_start
        self.min_exploration = min_exploration
        self.exploration_decay = exploration_decay
        self.device = torch.device('cuda' if torch.cuda.
                                   is_available() else 'cpu')
        self.target_update_count = 0
        self.update_target_interval = update_target_interval

    def save(self, save_dir, agent_model_name, qmixer_model_name):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        agent_model_path = os.path.join(save_dir, agent_model_name)
        qmixer_model_path = os.path.join(save_dir, agent_model_name)
        torch.save(self.alg.agent_model.state_dict(), agent_model_path)
        torch.save(self.alg.qmixer_model.state_dict(), qmixer_model_path)
        print('save model sucessfully!')

    def restore(self, save_dir, agent_model_name, qmixer_model_name):
        agent_model_path = os.path.join(save_dir, agent_model_name)
        qmixer_model_path = os.path.join(save_dir, agent_model_name)
        self.alg.agent_model.load_state_dict(torch.load(agent_model_path))
        self.alg.qmixer_model.load_state_dict(torch.load(qmixer_model_path))
        print('restore model sucessfully!')

    def reset_agent(self, batch_size=1):
        self.alg._init_hidden_states(batch_size)

    def sample(self, obs, available_actions):
        ''' sample actions via epsilon-greedy
        Args:
            obs (np.ndarray):               (n_agents, obs_shape)
            available_actions (np.ndarray): (n_agents, n_actions)
        Returns:
            actions (np.ndarray): sampled actions of agents
        '''
        epsilon = np.random.random()
        if epsilon > self.exploration:
            actions = self.predict(obs, available_actions)
        else:
            available_actions = torch.tensor(
                available_actions, dtype=torch.float32)
            actions = torch.distributions.Categorical(
                available_actions).sample().long().cpu().detach().numpy()
        self.exploration = max(self.min_exploration,
                               self.exploration - self.exploration_decay)
        return actions

    def predict(self, obs, available_actions):
        '''take greedy actions
        Args:
            obs (np.ndarray):               (n_agents, obs_shape)
            available_actions (np.ndarray): (n_agents, n_actions)
        Returns:
            actions (np.ndarray):           (n_agents, )
        '''
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        available_actions = torch.tensor(
            available_actions, dtype=torch.long, device=self.device)
        agents_q, self.alg.hidden_states = self.alg.predict_local_q(
            obs, self.alg.hidden_states)
        # mask unavailable actions
        agents_q[available_actions == 0] = -1e10
        actions = agents_q.max(dim=1)[1].detach().cpu().numpy()
        return actions

    def update_target(self):
        self.alg.target_agent_model.update(self.alg.agent_model)
        self.alg.target_qmixer_model.update(self.alg.qmixer_model)

    def learn(self, state_batch, actions_batch, reward_batch, terminated_batch,
              obs_batch, available_actions_batch, filled_batch):
        '''
        Args:
            state (np.ndarray):                   (batch_size, T, state_shape)
            actions (np.ndarray):                 (batch_size, T, n_agents)
            reward (np.ndarray):                  (batch_size, T, 1)
            terminated (np.ndarray):              (batch_size, T, 1)
            obs (np.ndarray):                     (batch_size, T, n_agents, obs_shape)
            available_actions_batch (np.ndarray): (batch_size, T, n_agents, n_actions)
            filled_batch (np.ndarray):            (batch_size, T, 1)
        Returns:
            mean_loss (float): train loss
            mean_td_error (float): train TD error
        '''
        if self.global_step % self.update_target_interval == 0:
            self.update_target()
            self.target_update_count += 1

        self.global_step += 1

        state_batch = torch.tensor(
            state_batch, dtype=torch.float32, device=self.device)
        actions_batch = torch.tensor(
            actions_batch, dtype=torch.long, device=self.device)
        reward_batch = torch.tensor(
            reward_batch, dtype=torch.float32, device=self.device)
        terminated_batch = torch.tensor(
            terminated_batch, dtype=torch.float32, device=self.device)
        obs_batch = torch.tensor(
            obs_batch, dtype=torch.float32, device=self.device)
        available_actions_batch = torch.tensor(
            available_actions_batch, dtype=torch.float32, device=self.device)
        filled_batch = torch.tensor(
            filled_batch, dtype=torch.float32, device=self.device)
        mean_loss, mean_td_error = self.alg.learn(
            state_batch, actions_batch, reward_batch, terminated_batch,
            obs_batch, available_actions_batch, filled_batch)
        return mean_loss, mean_td_error
