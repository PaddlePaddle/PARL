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
import copy
from copy import deepcopy
from paddle.fluid import layers
import paddle.fluid as fluid

__all__ = ['QMIX']


class QMIX(parl.Algorithm):
    def __init__(self, agent_model, qmixer_model, config):
        """ QMIX algorithm
        Args:
            agent_model (parl.Model): agents' local q network for decision making.
            qmixer_model (parl.Model): A mixing network which takes local q values as input
            config (dict): config of the algorithm.
        """
        self.agent_model = agent_model
        self.qmixer_model = qmixer_model
        self.target_agent_model = deepcopy(self.agent_model)
        self.target_qmixer_model = deepcopy(self.qmixer_model)

        self.lr = config['lr']
        self.gamma = config['gamma']
        self.double_q = config['double_q']
        self.n_agents = config['n_agents']
        self.n_actions = config['n_actions']
        self.obs_shape = config['obs_shape']
        self.batch_size = config['batch_size']
        self.episode_limit = config['episode_limit']
        self.rnn_hidden_dim = config['rnn_hidden_dim']
        self.clip_grad_norm = config['clip_grad_norm']
        assert isinstance(self.gamma, float)
        assert isinstance(self.lr, float)

    def predict_local_q(self, obs, hidden_state):
        ''' Predict local q values for each agent.
        Args:
            obs:           (n_agents, obs_shape)
            hidden_states: (n_agents, rnn_hidden_dim)
        Returns:
            self.agent_model(obs, hidden_state)
        '''
        return self.agent_model(obs, hidden_state)

    def learn(self, init_hidden_states, target_init_hidden_states, state_batch,
              actions_batch, reward_batch, terminated_batch, obs_batch,
              available_actions_batch, filled_batch):
        '''
        Args:
            init_hidden_states:       (batch_size, n_agents, rnn_hidden_dim)
            target_init_hidden_states:(batch_size, n_agents, rnn_hidden_dim)
            state_batch:              (batch_size, T, state_shape)
            actions_batch:            (batch_size, T, n_agents)
            reward_batch:             (batch_size, T, 1)
            terminated_batch:         (batch_size, T, 1)
            obs_batch:                (batch_size, T, n_agents, obs_shape)
            available_actions_batch:  (batch_size, T, n_agents, n_actions)
            filled_batch:             (batch_size, T, 1)
        Returns:
            loss (float): train loss
            td_erro (float): train TD error
        '''
        hidden_states = init_hidden_states
        target_hidden_states = target_init_hidden_states

        reward_batch = layers.slice(
            reward_batch, axes=[1], starts=[0], ends=[-1])
        actions_batch = layers.unsqueeze(
            layers.slice(actions_batch, axes=[1], starts=[0], ends=[-1]),
            axes=[-1])
        terminated_batch = layers.slice(
            terminated_batch, axes=[1], starts=[0], ends=[-1])
        filled_batch = layers.slice(
            filled_batch, axes=[1], starts=[0], ends=[-1])

        mask = (1 - filled_batch) * (1 - terminated_batch)

        local_qs = []
        target_local_qs = []
        for t in range(self.episode_limit):
            obs = obs_batch[:, t, :, :]
            obs = layers.reshape(
                obs, shape=(self.batch_size * self.n_agents, self.obs_shape))
            hidden_states = layers.reshape(
                hidden_states,
                shape=(self.batch_size * self.n_agents, self.rnn_hidden_dim))
            local_q, hidden_states = self.agent_model(obs, hidden_states)
            local_q = layers.reshape(
                local_q,
                shape=(self.batch_size, self.n_agents, self.n_actions))
            local_qs.append(local_q)

            target_hidden_states = layers.reshape(
                target_hidden_states,
                shape=(self.batch_size * self.n_agents, self.rnn_hidden_dim))
            target_local_q, target_hidden_states = self.target_agent_model(
                obs, target_hidden_states)
            target_local_q = layers.reshape(
                target_local_q,
                shape=(self.batch_size, self.n_agents, self.n_actions))
            target_local_qs.append(target_local_q)

        local_qs = layers.stack(local_qs, axis=1)
        target_local_qs = layers.stack(target_local_qs[1:], axis=1)

        actions_batch_one_hot = layers.one_hot(
            actions_batch, depth=self.n_actions)
        actions_batch_one_hot = layers.cast(
            actions_batch_one_hot, dtype='float32')
        sliced_local_qs = layers.slice(
            local_qs, axes=[1], starts=[0], ends=[-1])
        chosen_action_local_qs = layers.reduce_sum(
            layers.elementwise_mul(actions_batch_one_hot, sliced_local_qs),
            dim=-1)

        available_actions_batch = layers.cast(
            available_actions_batch, dtype='float32')
        action_mask = -1e10 * (1.0 - available_actions_batch)
        target_action_mask = layers.slice(
            action_mask, axes=[1], starts=[1], ends=[10000])
        target_action_zero_mask = layers.slice(
            available_actions_batch, axes=[1], starts=[1], ends=[10000])
        # mask unavailable actions
        target_local_qs = target_local_qs * target_action_zero_mask + target_action_mask

        if self.double_q:
            # mask unavailable actions
            masked_local_qs = local_qs * available_actions_batch + action_mask
            masked_local_qs.stop_gradient = True
            masked_local_qs = layers.slice(
                masked_local_qs, axes=[1], starts=[1], ends=[100000])
            cur_max_actions = layers.unsqueeze(
                layers.argmax(masked_local_qs, axis=-1), axes=-1)
            cur_max_actions_one_hot = layers.one_hot(
                cur_max_actions, depth=self.n_actions)
            cur_max_actions_one_hot = layers.cast(
                cur_max_actions_one_hot, dtype='float32')
        else:
            cur_max_actions = layers.unsqueeze(
                layers.argmax(target_local_qs, axis=-1), axes=-1)
            cur_max_actions_one_hot = layers.one_hot(
                cur_max_actions, depth=self.n_actions)
            cur_max_actions_one_hot = layers.cast(
                cur_max_actions_one_hot, dtype='float32')

        target_local_max_qs = layers.reduce_sum(
            layers.elementwise_mul(cur_max_actions_one_hot, target_local_qs),
            dim=-1)

        chosen_action_global_qs = self.qmixer_model(
            chosen_action_local_qs,
            layers.slice(state_batch, axes=[1], starts=[0], ends=[-1]))
        target_global_max_qs = self.target_qmixer_model(
            target_local_max_qs,
            layers.slice(state_batch, axes=[1], starts=[1], ends=[10000]))

        target = reward_batch + self.gamma * (
            1 - terminated_batch) * target_global_max_qs
        target.stop_gradient = True

        td_error = target - chosen_action_global_qs
        masked_td_error = td_error * mask
        mean_td_error = layers.reduce_sum(masked_td_error) / layers.reduce_sum(
            mask)

        loss = layers.reduce_sum(
            layers.square(masked_td_error)) / layers.reduce_sum(mask)
        if self.clip_grad_norm:
            clip = fluid.clip.GradientClipByGlobalNorm(
                clip_norm=self.clip_grad_norm)
            optimizer = fluid.optimizer.RMSPropOptimizer(
                learning_rate=self.lr, rho=0.99, epsilon=1e-6, grad_clip=clip)
        else:
            optimizer = fluid.optimizer.RMSPropOptimizer(
                learning_rate=self.lr, rho=0.99, epsilon=1e-6)

        optimizer.minimize(loss)
        return loss, mean_td_error

    def sync_target(self):
        '''sync weights of model to target model
        '''
        self.agent_model.sync_weights_to(self.target_agent_model)
        self.qmixer_model.sync_weights_to(self.target_qmixer_model)
