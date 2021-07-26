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
import paddle
from parl.utils.scheduler import LinearDecayScheduler


class AtariAgent(parl.Agent):
    """Agent of Atari env.

    Args:
        algorithm (`parl.Algorithm`): algorithm to be used in this agent.
        act_dim (int): action space dimension
        start_lr (float): starting learning rate
        total_step (int): total learning rate and epsilon decay steps
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

        obs = paddle.to_tensor(obs, dtype='float32')
        pred_q = self.alg.predict(obs).detach().numpy().squeeze()

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

        obs = paddle.to_tensor(obs, dtype='float32')
        act = paddle.to_tensor(act, dtype='int32')
        reward = paddle.to_tensor(reward, dtype='float32')
        next_obs = paddle.to_tensor(next_obs, dtype='float32')
        terminal = paddle.to_tensor(terminal, dtype='float32')

        loss = self.alg.learn(obs, act, reward, next_obs, terminal)

        # learning rate decay
        self.alg.optimizer.set_lr(max(self.lr_scheduler.step(1), self.lr_end))
        return loss.numpy()[0]
