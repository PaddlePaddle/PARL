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
import paddle
import numpy as np
from parl.utils import get_gpu_count


class CartpoleAgent(parl.Agent):
    """Agent of Cartpole env.

    Args:
        algorithm(parl.Algorithm): algorithm used to solve the problem.

    """

    def __init__(self, algorithm):
        super(CartpoleAgent, self).__init__(algorithm)
        gpu_count = get_gpu_count()
        if gpu_count > 0:
            self.place = paddle.CUDAPlace(0)
        else:
            self.place = paddle.CPUPlace()

    def sample(self, obs):
        """Sample an action when given an observation

        Args:
            obs(np.float32): shape of (obs_dim,)
        
        Returns:
            action(int)
        """
        obs = paddle.to_tensor(obs.astype(np.float32), place=self.place)
        prob = self.alg.predict(obs)
        prob = prob.numpy()
        action = np.random.choice(len(prob), 1, p=prob)[0]
        return action

    def predict(self, obs):
        """Predict an action when given an observation

        Args:
            obs(np.float32): shape of (obs_dim,)
        
        Returns:
            action(int)
        """
        obs = paddle.to_tensor(obs.astype(np.float32), place=self.place)
        prob = self.alg.predict(obs)
        action = prob.argmax().numpy()[0]
        return action

    def learn(self, obs, action, reward):
        """Update model with an episode data

        Args:
            obs(np.float32): shape of (batch_size, obs_dim)
            action(np.int64): shape of (batch_size)
            reward(np.float32): shape of (batch_size)
        
        Returns:
            loss(float)

        """
        action = np.expand_dims(action, axis=1)
        reward = np.expand_dims(reward, axis=1)

        obs = paddle.to_tensor(obs.astype(np.float32), place=self.place)
        action = paddle.to_tensor(action.astype(np.int32), place=self.place)
        reward = paddle.to_tensor(reward.astype(np.float32), place=self.place)

        loss = self.alg.learn(obs, action, reward)
        return loss.numpy()[0]
