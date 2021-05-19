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
import numpy as np
import paddle


class AtariAgent(parl.Agent):

    def __init__(self, algorithm, config):

        super().__init__(algorithm)

        self.global_update_step = 0
        self.explore_step = 0
        self.learning_step = 0
        self.update_target_step = config['update_target_step']
        self.act_dim = config['act_dim']
        self.ep_scheduler = LinearScheduler(config['ep_start'], config['ep_end'], config['ep_step'])
        self.curr_ep = self.ep_scheduler.step(self.explore_step)
        self.lr_scheduler = LinearScheduler(config['lr_start'], config['lr_end'], config['lr_step'])

    def sample(self, obs):

        explore = np.random.choice([True, False], p=[self.curr_ep, 1 - self.curr_ep])

        if explore:
            act = np.random.randint(self.act_dim)

        else:
            
            with paddle.no_grad():
                act = self.predict(obs)

        self.explore_step = self.explore_step + 1
        self.curr_ep = self.ep_scheduler.step(self.explore_step)

        return act

    def predict(self, obs):
        """Predict an action when given an observation

        Args:
            obs(np.float32): shape of (obs_dim,)

        Returns:
            act(int): action
        """
        if obs.ndim == 3:
            obs = np.expand_dims(obs, axis=0)

        obs = paddle.to_tensor(obs, dtype='float32')
        pred_q = self.alg.predict(obs).numpy().squeeze()
        best_actions = np.where(pred_q == pred_q.max())[0]
        act = np.random.choice(best_actions)

        return act

    # def sample(self, obs):
    #     sample = np.random.random()
    #     if sample < self.curr_ep:
    #         act = np.random.randint(self.act_dim)
    #     else:
    #         if np.random.random() < 0.01:
    #             act = np.random.randint(self.act_dim)
    #         else:

    #             act = self.predict(obs)

    #     self.explore_step = self.explore_step + 1
    #     self.curr_ep = self.ep_scheduler.step(self.explore_step)
        
    #     return act

    # def predict(self, obs):
    #     obs = np.expand_dims(obs, axis=0)
    #     obs = paddle.to_tensor(obs, dtype='float32')
    #     pred_Q = self.alg.predict(obs).numpy().squeeze(axis=0)
    #     act = np.argmax(pred_Q)
    #     return act

    def learn(self, obs, act, reward, next_obs, terminal):

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
        self.learning_step = self.learning_step + 1
        self.alg.optimizer.set_lr(self.lr_scheduler.step(self.learning_step)) 
        
        return loss.numpy()[0]


class LinearScheduler:

    def __init__(self, val_begin, val_end, n_steps):
        """
        Args:
            val_begin: initial value
            val_end: end value
            nsteps: number of steps between the two values
        """

        self.val_begin = val_begin
        self.val_end = val_end
        self.n_steps = n_steps

    def step(self, t):
        """
        get curr_val

        Args:
            t: int
                frame number
        """
        if t <= self.n_steps:
            return self.val_begin + t * (self.val_end - self.val_begin) / self.n_steps

        else:
            return self.val_end



