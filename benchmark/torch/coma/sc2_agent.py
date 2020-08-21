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

import numpy as np
import parl
import torch
from torch.distributions import Categorical

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Agents(parl.Agent):
    def __init__(self, algorithm, config):
        self.n_actions = config['n_actions']
        self.n_agents = config['n_agents']
        self.state_shape = config['state_shape']
        self.obs_shape = config['obs_shape']

        self.config = config
        self.train_steps = 0
        self.rnn_h = None
        super(Agents, self).__init__(algorithm)
        print('Init all agents')

    def init_hidden(self):
        """ function: init a hidden tensor for every agent at the begging of every episode
            self.rnn_h: rnn hidden state, shape (n_agents, hidden_size)
        """
        self.rnn_h = self.alg.init_hidden(1)[0]

    def predict(self, obs, rnn_h_in):
        """input:
                obs: obs + last_action + agent_id, shape: (1, obs_shape + n_actions + n_agents)
                rnn_h_in: rnn's hidden input
            output:
                prob: output of actor, shape: (1, n_actions)
                rnn_h_out: rnn's hidden output
        """
        obs = np.expand_dims(obs, 0)
        obs = torch.tensor(obs, dtype=torch.float32).to(device)
        prob, rnn_h_out = self.alg.predict(obs, rnn_h_in)
        return prob, rnn_h_out

    def sample(self,
               obs,
               last_action,
               agent_id,
               avail_actions,
               epsilon,
               test=False):
        """input:
                obs (array): agent i's obs
                last_action (int): agent i's last action
                agent_id (int): agent index
                avail_actions (one_hot): available actions
                epsilon (float): e_greed discount
                test (bool): train or test
            output:
                action: int
                prob: probability of every action, float, 0 ~ 1
        """
        obs = obs.copy()
        # make obs: obs + agent's last action(one_hot) + agent's id(one_hot)
        last_act_one_hot = np.zeros(self.n_actions)
        last_act_one_hot[last_action] = 1.
        id_one_hot = np.zeros(self.n_agents)
        id_one_hot[agent_id] = 1.
        obs = np.hstack((obs, last_act_one_hot))
        obs = np.hstack((obs, id_one_hot))

        # predict action prob
        prob, self.rnn_h[agent_id] = self.predict(obs, self.rnn_h[agent_id])

        # add noise
        avail_actions = torch.tensor(
            avail_actions, dtype=torch.float32).unsqueeze(0).to(
                device)  # shape: (1, n_actions)
        action_num = avail_actions.sum()  # how many actions are available
        prob = ((1 - epsilon) * prob +
                torch.ones_like(prob) * epsilon / action_num)
        prob[avail_actions == 0] = 0.0  # set avail action

        # choose action
        if epsilon == 0 or test:
            action = torch.argmax(prob)
        else:
            action = Categorical(prob).sample().long()
        return action.cpu()

    def _get_actor_inputs(self, batch):
        """ o(t), u(t-1)_a, agent_id
        """
        obs = batch['o']
        u_onehot = batch['u_onehot']
        u_onehot_last = np.zeros_like(u_onehot)
        u_onehot_last[:, 1:] = u_onehot[:, :-1]
        ep_num = batch['o'].shape[0]
        tr_num = batch['o'].shape[1]

        actor_inputs = []
        for agent_id in range(self.n_agents):
            obs_a = obs[:, :, agent_id]
            u_a_onehot_last = u_onehot_last[:, :, agent_id]
            id_onehot = np.zeros((ep_num, tr_num, self.n_agents))
            id_onehot[:, :, agent_id] = 1.
            # actor inputs: obs + agent's last action(one_hot) + agent's id(one_hot)
            a_inputs = np.concatenate((obs_a, u_a_onehot_last, id_onehot),
                                      axis=2)
            # a_inpits shape (ep_num, tr_num, actor_input_dim)
            actor_inputs.append(a_inputs)

        actor_inputs = np.stack(
            actor_inputs,
            axis=2)  # shape (ep_num, tr_num, n_agents, actor_input_dim)
        return actor_inputs

    def _get_critic_inputs(self, batch):
        """ o(t)_a, s(t), u(t)_-a, u(t-1), agent_id
        """
        ep_num = batch['o'].shape[0]
        tr_num = batch['o'].shape[1]

        # o, o_next, state, state_next
        o = batch['o']  # shape (ep_num, tr_num, n_agents, obs_shape)
        o_next = np.zeros_like(o)
        o_next[:, :-1] = o[:, 1:]
        s = batch['s']  # shape (ep_num, tr_num, state_shape)
        s_next = np.zeros_like(s)
        s_next[:, :-1] = s[:, 1:]
        # u_onehot, u_onehot_last shape (ep_num, tr_num, n_agents, n_actions)
        u_onehot = batch['u_onehot']
        u_onehot_next = np.zeros_like(u_onehot)
        u_onehot_next[:, :-1] = u_onehot[:, 1:]
        u_onehot_last = np.zeros_like(u_onehot)
        u_onehot_last[:, 1:] = u_onehot[:, :-1]

        critic_inputs = []
        critic_inputs_next = []
        for agent_id in range(self.n_agents):
            # get o(t)_a, s(t)
            o_a = o[:, :, agent_id]  # shape (ep_num, tr_num, obs_shape)
            o_a_next = o_next[:, :, agent_id]
            s_a = s  # shape (ep_num, tr_num, state_shape)
            s_a_next = s_next
            # get u(t-1)  shape (ep_num, tr_num, n_agents * n_actions)
            u_all_onehot = u_onehot.reshape((ep_num, tr_num,
                                             self.n_agents * self.n_actions))
            u_all_onehot_next = u_onehot_next.reshape(
                (ep_num, tr_num, self.n_agents * self.n_actions))
            u_all_onehot_last = u_onehot_last.reshape(
                (ep_num, tr_num, self.n_agents * self.n_actions))
            # get u(t)_-a,   set 0 to mask action, shape (ep_num, tr_num, n_agents * n_actions)
            u_not_a_onehot = u_all_onehot.copy()
            u_not_a_onehot_next = u_all_onehot_next.copy()
            m_s = agent_id * self.n_actions  # mask start flag
            m_e = (agent_id + 1) * self.n_actions  # mask end flag
            u_not_a_onehot[:, :, m_s:m_e] = 0
            u_not_a_onehot_next[:, :, m_s:m_e] = 0
            # get id onehot, shape (ep_num, tr_num, n_agents)
            id_onehot = np.zeros((ep_num, tr_num, self.n_agents))
            id_onehot[:, :, agent_id] = 1.

            # input:      o,      s,      u_-a,      u_last, agent_id
            # input_next: o_next, s_next, u_-a_next, u,      agent_id
            # shape (ep_num, tr_num, critic_input_dim)
            c_inputs = np.concatenate(
                (o_a, s_a, u_not_a_onehot, u_all_onehot_last, id_onehot),
                axis=2)
            c_inputs_next = np.concatenate(
                (o_a_next, s_a_next, u_not_a_onehot_next, u_all_onehot,
                 id_onehot),
                axis=2)
            critic_inputs.append(c_inputs)
            critic_inputs_next.append(c_inputs_next)
        critic_inputs = np.stack(critic_inputs, axis=2)
        critic_inputs_next = np.stack(critic_inputs_next, axis=2)
        # shape (ep_num, tr_num, n_agents, critic_input_dim)
        return critic_inputs, critic_inputs_next

    def _get_avail_transitions_num(self, isover_batch):
        """ input:
                isover_batch: shape (ep_num, tr_num, 1)
            output: 
                max_tr_num: max avail transitions number in all episodes
        """
        ep_num = isover_batch.shape[0]
        max_tr_num = 0
        for ep_id in range(ep_num):
            for tr_id in range(self.config['episode_limit']):
                if isover_batch[ep_id, tr_id, 0] == 1:
                    if tr_id + 1 >= max_tr_num:
                        max_tr_num = tr_id + 1
                    break
        return max_tr_num

    def learn(self, batch, epsilon=None):
        """ input:
                batch: dict(o, s, u, r, u_onehot, avail_u, padded, isover)
                epsilon: e-greedy discount
        """
        # different episode has different avail transition length
        tr_num = self._get_avail_transitions_num(batch['isover'])
        for key in batch.keys():
            # cut batch data's episode length
            batch[key] = batch[key][:, :tr_num]

        # get actor input and critic input
        batch['actor_inputs'] = self._get_actor_inputs(batch)
        batch['critic_inputs'], batch[
            'critic_inputs_next'] = self._get_critic_inputs(batch)

        # change batch data to torch tensor
        for key in batch.keys():
            if key == 'u':
                batch[key] = torch.tensor(
                    batch[key], dtype=torch.long).to(device)
            else:
                batch[key] = torch.tensor(
                    batch[key], dtype=torch.float32).to(device)

        self.alg.learn(batch, epsilon)

        if self.train_steps > 0 and self.train_steps % self.config[
                'target_update_cycle'] == 0:
            self.alg.sync_target()
        self.train_steps += 1
