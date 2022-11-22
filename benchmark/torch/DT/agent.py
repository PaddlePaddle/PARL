#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from data_loader import DataLoader


class DTAgent(parl.Agent):
    def __init__(self, algorithm, config):
        super(DTAgent, self).__init__(algorithm)
        self.dataset = None
        self.config = config

    def predict(self, states, actions, rewards, returns_to_go, timesteps,
                **kwargs):
        action = self.alg.predict(states, actions, rewards, returns_to_go,
                                  timesteps)
        actions[-1] = action
        action = action.detach().cpu().numpy()
        return action

    def learn(self):
        batch_data = self.dataset.get_batch(self.config['batch_size'])
        loss = self.alg.learn(*batch_data)
        loss = loss.detach().cpu().item()
        return loss

    def load_data(self, dataset_path):
        config = self.config
        self.dataset = DataLoader(dataset_path, config['pct_traj'],
                                  config['max_ep_len'], config['rew_scale'])
        return self.dataset.state_mean, self.dataset.state_std
