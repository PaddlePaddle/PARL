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
from parl import layers
import paddle.fluid as fluid
import numpy as np
from utils import AvailableActionsSampler
from parl.utils import machine_info


class QMixAgent(parl.Agent):
    def __init__(self, algorithm, config):
        self.global_step = 0
        self.n_agents = config['n_agents']
        self.n_actions = config['n_actions']
        self.obs_shape = config['obs_shape']
        self.batch_size = config['batch_size']
        self.state_shape = config['state_shape']
        self.episode_limit = config['episode_limit']
        self.exploration = config['exploration_start']
        self.rnn_hidden_dim = config['rnn_hidden_dim']
        self.min_exploration = config['min_exploration']
        self.exploration_decay = config['exploration_decay']
        self.update_target_interval = config['update_target_interval']
        self.target_update_count = 0
        super(QMixAgent, self).__init__(algorithm)

    def reset_agent(self):
        '''Generate GRU's initial hidden states for prediction (batch_size=1)
        '''
        self.last_hidden_states = np.zeros(
            (self.n_agents, self.rnn_hidden_dim), dtype='float32')

    def _get_hidden_states(self):
        ''' Generate GRU's initial hidden states for learning (batch_size=batch_size)
        '''
        init_hidden_states = np.zeros(
            (self.batch_size, self.n_agents, self.rnn_hidden_dim),
            dtype='float32')
        target_init_hidden_states = np.zeros(
            (self.batch_size, self.n_agents, self.rnn_hidden_dim),
            dtype='float32')
        return init_hidden_states, target_init_hidden_states

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):
            last_hidden_states = fluid.data(
                name='last_hidden_states',
                shape=[self.n_agents, self.rnn_hidden_dim],
                dtype='float32')
            obs = fluid.data(
                name='obs',
                shape=[self.n_agents, self.obs_shape],
                dtype='float32')
            self.agents_q, self.current_hidden_states = self.alg.predict_local_q(
                obs, last_hidden_states)

        with fluid.program_guard(self.learn_program):
            init_hidden_states = fluid.data(
                name='init_hidden_states',
                shape=[self.batch_size, self.n_agents, self.rnn_hidden_dim],
                dtype='float32')
            target_init_hidden_states = fluid.data(
                name='target_init_hidden_states',
                shape=[self.batch_size, self.n_agents, self.rnn_hidden_dim],
                dtype='float32')
            state_batch = fluid.data(
                name='state_batch',
                shape=[self.batch_size, self.episode_limit, self.state_shape],
                dtype='float32')
            actions_batch = fluid.data(
                name='actions_batch',
                shape=[self.batch_size, self.episode_limit, self.n_agents],
                dtype='long')
            reward_batch = fluid.data(
                name='reward_batch',
                shape=[self.batch_size, self.episode_limit, 1],
                dtype='float32')
            terminated_batch = fluid.data(
                name='terminated_batch',
                shape=[self.batch_size, self.episode_limit, 1],
                dtype='float32')
            obs_batch = fluid.data(
                name='obs_batch',
                shape=[
                    self.batch_size, self.episode_limit, self.n_agents,
                    self.obs_shape
                ],
                dtype='float32')
            available_actions_batch = fluid.data(
                name='available_actions_batch',
                shape=[
                    self.batch_size, self.episode_limit, self.n_agents,
                    self.n_actions
                ],
                dtype='long')
            filled_batch = fluid.data(
                name='filled_batch',
                shape=[self.batch_size, self.episode_limit, 1],
                dtype='float32')
            self.loss, self.mean_td_error, = self.alg.learn(
                init_hidden_states, target_init_hidden_states, state_batch,
                actions_batch, reward_batch, terminated_batch, obs_batch,
                available_actions_batch, filled_batch)

    def sample(self, obs, available_actions):
        ''' Sample actions via epsilon-greedy.
        Args:
            obs (np.ndarray):               (n_agents, obs_shape)
            available_actions (np.ndarray): (n_agents, n_actions)
        Returns:
            actions (np.ndarray):           (n_agents, )
        '''
        epsilon = np.random.random()
        if epsilon > self.exploration:
            actions = self.predict(obs, available_actions)
        else:
            actions = AvailableActionsSampler(available_actions).sample()
        self.exploration = max(self.min_exploration,
                               self.exploration - self.exploration_decay)
        return actions

    def predict(self, obs, available_actions):
        ''' take greedy actions
        Args:
            obs (np.ndarray):               (n_agents, obs_shape)
            available_actions (np.ndarray): (n_agents, n_actions)
        Returns:
            actions (np.ndarray):           (n_agents, )
        '''
        feed = {
            'last_hidden_states': self.last_hidden_states,
            'obs': obs,
        }
        agents_q, self.last_hidden_states = self.fluid_executor.run(
            self.pred_program,
            feed=feed,
            fetch_list=[self.agents_q, self.current_hidden_states])
        agents_q[available_actions == 0] = -1e10
        actions = np.argmax(agents_q, axis=1)
        return actions

    def learn(self, state_batch, actions_batch, reward_batch, terminated_batch,
              obs_batch, available_actions_batch, filled_batch):
        '''
        Args:
            init_hidden_states (np.ndarray):       (batch_size, n_agents, rnn_hidden_dim)
            target_init_hidden_states (np.ndarray):(batch_size, n_agents, rnn_hidden_dim)
            state_batch (np.ndarray):              (batch_size, T, state_shape)
            actions_batch (np.ndarray):            (batch_size, T, n_agents)
            reward_batch (np.ndarray):             (batch_size, T, 1)
            terminated_batch (np.ndarray):         (batch_size, T, 1)
            obs_batch (np.ndarray):                (batch_size, T, n_agents, obs_shape)
            available_actions_batch (np.ndarray):  (batch_size, T, n_agents, n_actions)
            filled_batch (np.ndarray):             (batch_size, T, 1)
        Returns:
            mean_loss (float): train loss
            mean_td_error (float): train TD error
        '''
        init_hidden_states, target_init_hidden_states = self._get_hidden_states(
        )

        if self.global_step % self.update_target_interval == 0:
            self.alg.sync_target()
            self.target_update_count += 1

        self.global_step += 1

        feed = {
            'init_hidden_states': init_hidden_states,
            'target_init_hidden_states': target_init_hidden_states,
            'state_batch': state_batch,
            'actions_batch': actions_batch,
            'reward_batch': reward_batch,
            'terminated_batch': terminated_batch,
            'obs_batch': obs_batch,
            'available_actions_batch': available_actions_batch,
            'filled_batch': filled_batch,
        }
        mean_loss, mean_td_error = self.fluid_executor.run(
            self.learn_program,
            feed=feed,
            fetch_list=[self.loss, self.mean_td_error])
        mean_loss = mean_loss[0]
        mean_td_error = mean_td_error[0]
        return mean_loss, mean_td_error
