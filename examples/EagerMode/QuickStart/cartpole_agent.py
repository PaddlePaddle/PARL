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

import numpy as np
import paddle.fluid as fluid
from parl.utils import machine_info


class CartpoleAgent(object):
    def __init__(
            self,
            alg,
            obs_dim,
            act_dim,
    ):
        self.alg = alg
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def sample(self, obs):
        obs = np.expand_dims(obs, axis=0)
        act_prob = self.alg.predict(obs).numpy()
        act_prob = np.squeeze(act_prob, axis=0)
        act = np.random.choice(self.act_dim, p=act_prob)
        return act

    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0)
        act_prob = self.alg.predict(obs).numpy()
        act_prob = np.squeeze(act_prob, axis=0)
        act = np.argmax(act_prob)
        return act

    def learn(self, obs, act, reward):
        act = np.expand_dims(act, axis=-1)
        reward = np.expand_dims(reward, axis=-1)
        cost = self.alg.learn(obs, act, reward)
        return cost
