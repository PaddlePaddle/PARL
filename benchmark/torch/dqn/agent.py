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

import parl
import numpy as np
import torch
from parl.utils.scheduler import LinearDecayScheduler


class AtariAgent(parl.Agent):
    """Agent of Atari env.

    Args:
        algorithm (`parl.Algorithm`): algorithm to be used in this agent.
        act_dim (int): action space dimension
        total_step (int): total epsilon decay steps
        start_lr (float): initial learning rate
        update_target_step (int): target network update frequency
    """

    def __init__(self, algorithm, act_dim, start_lr, total_step,
                 update_target_step):
        super().__init__(algorithm)
        self.global_update_step = 0
        self.update_target_step = update_target_step
        self.act_dim = act_dim
        self.curr_ep = 1
        self.ep_end = 0.1
        self.lr_end = 0.00001
        self.device = torch.device('cuda' if torch.cuda.
                                   is_available() else 'cpu')

        self.ep_scheduler = LinearDecayScheduler(1, total_step)
        self.lr_scheduler = LinearDecayScheduler(start_lr, total_step)

    def sample(self, obs):
        """Sample an action when given an observation, base on the current epsilon value, 
        either a greedy action or a random action will be returned.

        Args:
            obs (np.float32): shape of (3, 84, 84) or (1, 3, 84, 84), current observation

        Returns:
            act (int): action
        """
        explore = np.random.choice([True, False],
                                   p=[self.curr_ep, 1 - self.curr_ep])
        if explore:
            act = np.random.randint(self.act_dim)
        else:
            act = self.predict(obs)

        self.curr_ep = max(self.ep_scheduler.step(1), self.ep_end)
        return act

    def predict(self, obs):
        """Predict an action when given an observation, a greedy action will be returned.

        Args:
            obs (np.float32): shape of (3, 84, 84) or (1, 3, 84, 84), current observation

        Returns:
            act(int): action
        """
        if obs.ndim == 3:  # if obs is 3 dimensional, we need to expand it to have batch_size = 1
            obs = np.expand_dims(obs, axis=0)

        obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        pred_q = self.alg.predict(obs).cpu().detach().numpy().squeeze()

        best_actions = np.where(pred_q == pred_q.max())[0]
        act = np.random.choice(best_actions)
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        """Update model with an episode data

        Args:
            obs (np.float32): shape of (batch_size, obs_dim)
            act (np.int32): shape of (batch_size)
            reward (np.float32): shape of (batch_size)
            next_obs (np.float32): shape of (batch_size, obs_dim)
            terminal (np.float32): shape of (batch_size)

        Returns:
            loss (float)
        """
        if self.global_update_step % self.update_target_step == 0:
            self.alg.sync_target()

        self.global_update_step += 1

        reward = np.clip(reward, -1, 1)
        act = np.expand_dims(act, axis=-1)
        reward = np.expand_dims(reward, axis=-1)
        terminal = np.expand_dims(terminal, axis=-1)

        obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        next_obs = torch.tensor(
            next_obs, dtype=torch.float, device=self.device)
        act = torch.tensor(act, dtype=torch.long, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float, device=self.device)
        terminal = torch.tensor(
            terminal, dtype=torch.float, device=self.device)

        loss = self.alg.learn(obs, act, reward, next_obs, terminal)

        # learning rate decay
        for param_group in self.alg.optimizer.param_groups:
            param_group['lr'] = max(self.lr_scheduler.step(1), self.lr_end)

        return loss
