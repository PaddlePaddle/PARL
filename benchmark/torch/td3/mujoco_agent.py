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

import parl
import torch
import numpy as np


class MujocoAgent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim, expl_noise=0.1):
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        super(MujocoAgent, self).__init__(algorithm)

        self.device = torch.device("cuda" if torch.cuda.
                                   is_available() else "cpu")

        self.alg.sync_target(decay=0)
        self.expl_noise = expl_noise

    def sample(self, obs):
        action = np.random.normal(
            self.predict(np.array(obs)), self.expl_noise).clip(-1, 1)
        return action

    def predict(self, obs):
        obs = torch.FloatTensor(obs.reshape(1, -1)).to(self.device)
        return self.alg.predict(obs).cpu().detach().numpy().flatten()

    def learn(self, obs, act, reward, next_obs, terminal):
        terminal = np.expand_dims(terminal, -1)
        reward = np.expand_dims(reward, -1)

        obs = torch.FloatTensor(obs).to(self.device)
        act = torch.FloatTensor(act).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        terminal = torch.FloatTensor(terminal).to(self.device)
        self.alg.learn(obs, act, reward, next_obs, terminal)
