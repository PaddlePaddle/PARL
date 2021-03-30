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

import paddle
import parl
import numpy as np
from utils import AvailableActionsSampler
import os


class QMixAgent(parl.Agent):
    def __init__(self, algorithm, exploration_start, min_exploration,
                 exploration_decay, update_target_interval):
        self.alg = algorithm
        self.global_step = 0
        self.exploration = exploration_start
        self.min_exploration = min_exploration
        self.exploration_decay = exploration_decay
        self.target_update_count = 0
        self.update_target_interval = update_target_interval

    def save(self, save_dir, agent_model_name, qmixer_model_name):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        agent_model_path = os.path.join(save_dir, agent_model_name)
        qmixer_model_path = os.path.join(save_dir, qmixer_model_name)
        paddle.save(self.alg.agent_model.state_dict(), agent_model_path)
        paddle.save(self.alg.qmixer_model.state_dict(), qmixer_model_path)
        print('save model successfully!')

    def restore(self, save_dir, agent_model_name, qmixer_model_name):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        agent_model_path = os.path.join(save_dir, agent_model_name)
        qmixer_model_path = os.path.join(save_dir, qmixer_model_name)
        self.alg.agent_model.set_state_dict(paddle.load(agent_model_path))
        self.alg.qmixer_model.set_state_dict(paddle.load(qmixer_model_path))
        print('restore model successfully!')

    def reset_agent(self, batch_size=1):
        self.alg._init_hidden_states(batch_size)

    def sample(self, obs, available_actions):
        """ sample actions via epsilon-greedy
        Args:
            obs (np.ndarray):                (n_agents, obs_shape)
            available_actions (np.ndarray):  (n_agents, n_actions)
        Returns:
            actions (np.ndarray):            (n_agents, )
        """
        epsilon = np.random.random()
        if epsilon > self.exploration:
            actions = self.predict(obs, available_actions)
        else:
            actions = AvailableActionsSampler(available_actions).sample()
        self.exploration = max(self.min_exploration,
                               self.exploration - self.exploration_decay)
        return actions

    def predict(self, obs, available_actions):
        """ take greedy actions
        Args:
            obs (np.ndarray):                (n_agents, obs_shape)
            available_actions (np.ndarray):  (n_agents, n_actions)
        Returns:
            actions (np.ndarray):            (n_agents, )
        """
        obs = paddle.to_tensor(obs, dtype='float32')
        available_actions = paddle.to_tensor(available_actions, dtype='int32')
        agents_q, self.alg.hidden_states = self.alg.predict_local_q(
            obs, self.alg.hidden_states)
        # mask unavailable actions
        unavailable_actions_mask = (available_actions == 0).cast('float32')
        agents_q -= 1e8 * unavailable_actions_mask
        actions = paddle.argmax(agents_q, axis=-1).detach().cpu().numpy()
        return actions

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
            self.alg.sync_target()
            self.target_update_count += 1

        self.global_step += 1

        state_batch = paddle.to_tensor(state_batch, dtype='float32')
        actions_batch = paddle.to_tensor(actions_batch, dtype='int64')
        reward_batch = paddle.to_tensor(reward_batch, dtype='float32')
        terminated_batch = paddle.to_tensor(terminated_batch, dtype='float32')
        obs_batch = paddle.to_tensor(obs_batch, dtype='float32')
        available_actions_batch = paddle.to_tensor(
            available_actions_batch, dtype='int64')
        filled_batch = paddle.to_tensor(filled_batch, dtype='float32')
        mean_loss, mean_td_error = self.alg.learn(
            state_batch, actions_batch, reward_batch, terminated_batch,
            obs_batch, available_actions_batch, filled_batch)
        return mean_loss, mean_td_error
