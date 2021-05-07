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
import paddle
import numpy as np


class MujocoAgent(parl.Agent):
    def __init__(self, algorithm, act_dim, expl_noise=0.1):
        super(MujocoAgent, self).__init__(algorithm)

        self.alg.sync_target(decay=0)
        self.expl_noise = expl_noise
        self.action_dim = act_dim

    def predict(self, obs):
        obs = paddle.to_tensor(obs.reshape(1, -1), dtype='float32')
        action = self.alg.predict(obs)
        action_numpy = action.cpu().numpy()[0]
        return action_numpy

    def sample(self, obs):
        action_numpy = self.predict(obs)
        action_noise = np.random.normal(
            0, self.expl_noise, size=self.action_dim)
        action = (action_numpy + action_noise).clip(-1, 1)
        return action

    def learn(self, obs, action, reward, next_obs, terminal):
        terminal = np.expand_dims(terminal, -1)
        reward = np.expand_dims(reward, -1)

        obs = paddle.to_tensor(obs, dtype='float32')
        action = paddle.to_tensor(action, dtype='float32')
        reward = paddle.to_tensor(reward, dtype='float32')
        next_obs = paddle.to_tensor(next_obs, dtype='float32')
        terminal = paddle.to_tensor(terminal, dtype='float32')
        critic_loss = self.alg.learn(obs, action, reward, next_obs, terminal)
        return critic_loss
