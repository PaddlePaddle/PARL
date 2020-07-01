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

import parl
import torch
import numpy as np


class CartpoleAgent(parl.Agent):
    """Agent of Cartpole env.

    Args:
        algorithm(parl.Algorithm): algorithm used to solve the problem.

    """

    def __init__(self, algorithm):
        super(CartpoleAgent, self).__init__(algorithm)
        self.device = torch.device("cuda" if torch.cuda.
                                   is_available() else "cpu")

    def sample(self, obs):
        """Sample an action when given an observation

        Args:
            obs(np.float32): shape of (obs_dim,)
        
        Returns:
            action(int)
        """
        obs = torch.tensor(obs, device=self.device, dtype=torch.float)
        prob = self.alg.predict(obs).cpu()
        prob = prob.data.numpy()
        action = np.random.choice(len(prob), 1, p=prob)[0]
        return action

    def predict(self, obs):
        """Predict an action when given an observation

        Args:
            obs(np.float32): shape of (obs_dim,)
        
        Returns:
            action(int)
        """
        obs = torch.tensor(obs, device=self.device, dtype=torch.float)
        prob = self.alg.predict(obs)
        _, action = prob.max(-1)
        return action.item()

    def learn(self, obs, action, reward):
        """Update model with an episode data

        Args:
            obs(np.float32): shape of (batch_size, obs_dim)
            action(np.int64): shape of (batch_size)
            reward(np.float32): shape of (batch_size)
        
        Returns:
            loss(float)

        """
        obs = torch.tensor(obs, device=self.device, dtype=torch.float)
        action = torch.tensor(action, device=self.device, dtype=torch.long)
        reward = torch.tensor(reward, device=self.device, dtype=torch.float)

        loss = self.alg.learn(obs, action, reward)
        return loss.item()
