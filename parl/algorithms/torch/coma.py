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

import torch
import os
from copy import deepcopy
import parl
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

__all__ = ['COMA']


class COMA(parl.Algorithm):
    def __init__(self,
                 model,
                 n_actions,
                 n_agents,
                 grad_norm_clip=None,
                 actor_lr=None,
                 critic_lr=None,
                 gamma=None,
                 td_lambda=None):
        """  COMA algorithm
        
        Args:
            model (parl.Model): forward network of actor and critic.
            n_actions (int): action dim for each agent
            n_agents (int): agents number
            grad_norm_clip (int or float): gradient clip, prevent gradient explosion
            actor_lr (float): actor network learning rate
            critic_lr (float): critic network learning rate
            gamma (float):  discounted factor for reward computation
            td_lambda (float): lambda of td-lambda return
        """
        assert isinstance(n_actions, int)
        assert isinstance(n_agents, int)
        assert isinstance(grad_norm_clip, int) or isinstance(
            grad_norm_clip, float)
        assert isinstance(actor_lr, float)
        assert isinstance(critic_lr, float)
        assert isinstance(gamma, float)
        assert isinstance(td_lambda, float)

        self.n_actions = n_actions
        self.n_agents = n_agents
        self.grad_norm_clip = grad_norm_clip
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.td_lambda = td_lambda

        self.model = model.to(device)
        self.target_model = deepcopy(model).to(device)

        self.sync_target()

        self.actor_parameters = list(self.model.get_actor_params())
        self.critic_parameters = list(self.model.get_critic_params())

        self.critic_optimizer = torch.optim.RMSprop(
            self.critic_parameters, lr=self.critic_lr)
        self.actor_optimizer = torch.optim.RMSprop(
            self.actor_parameters, lr=self.actor_lr)

        self.train_rnn_h = None

    def init_hidden(self, ep_num):
        """ function: init a hidden tensor for every agent
            input: 
                ep_num: How many episodes are included in a batch of data
            output:
                rnn_h: rnn hidden state, shape (ep_num, n_agents, hidden_size)
        """
        assert hasattr(self.model.actor_model, 'init_hidden'), \
            "actor must have rnn structure and has method 'init_hidden' to make hidden states"
        rnn_h = self.model.actor_model.init_hidden().unsqueeze(0).expand(
            ep_num, self.n_agents, -1)
        return rnn_h

    def predict(self, obs, rnn_h_in):
        """input:
                obs: obs + last_action + agent_id, shape: (1, obs_shape + n_actions + n_agents)
                rnn_h_in: rnn's hidden input
            output:
                prob: output of actor, shape: (1, n_actions)
                rnn_h_out: rnn's hidden output
        """
        with torch.no_grad():
            policy_logits, rnn_h_out = self.model.policy(
                obs, rnn_h_in)  # input obs shape [1, 42]
            prob = torch.nn.functional.softmax(
                policy_logits, dim=-1)  # shape [1, 9]
        return prob, rnn_h_out

    def _get_critic_output(self, batch):
        """ input:
                batch: dict(o, s, u, r, u_onehot, avail_u, padded, isover, actor_inputs, critic_inputs)
            output:
                q_evals and q_targets: shape (ep_num, tr_num, n_agents, n_actions)
        """
        ep_num = batch['r'].shape[0]
        tr_num = batch['r'].shape[1]
        critic_inputs = batch['critic_inputs']
        critic_inputs_next = batch['critic_inputs_next']

        critic_inputs = critic_inputs.reshape((ep_num * tr_num * self.n_agents,
                                               -1))
        critic_inputs_next = critic_inputs.reshape(
            (ep_num * tr_num * self.n_agents, -1))

        q_evals = self.model.value(critic_inputs)
        q_targets = self.model.value(critic_inputs_next)

        q_evals = q_evals.reshape((ep_num, tr_num, self.n_agents, -1))
        q_targets = q_targets.reshape((ep_num, tr_num, self.n_agents, -1))
        return q_evals, q_targets

    def _get_actor_output(self, batch, epsilon):
        """ input:
                batch: dict(o, s, u, r, u_onehot, avail_u, padded, isover, actor_inputs, critic_inputs)
                epsilon: noise discount factor
            output:
                action_prob: probability of actions, shape (ep_num, tr_num, n_agents, n_actions)
        """
        ep_num = batch['r'].shape[0]
        tr_num = batch['r'].shape[1]
        avail_actions = batch['avail_u']
        actor_inputs = batch['actor_inputs']
        action_prob = []
        for tr_id in range(tr_num):
            inputs = actor_inputs[:,
                                  tr_id]  # shape (ep_num, n_agents, actor_input_dim)
            inputs = inputs.reshape(
                (-1, inputs.shape[-1]))  # shape (-1, actor_input_dim)
            policy_logits, self.train_rnn_h = self.model.policy(
                inputs, self.train_rnn_h)
            # policy_logits shape from (-1, n_actions) to (ep_num, n_agents, n_actions)
            policy_logits = policy_logits.view(ep_num, self.n_agents, -1)
            prob = torch.nn.functional.softmax(policy_logits, dim=-1)
            action_prob.append(prob)

        action_prob = torch.stack(
            action_prob,
            dim=1).to(device)  # shape: (ep_num, tr_num, n_agents, n_actions)
        action_num = avail_actions.sum()  # how many actions are available
        action_prob = ((1 - epsilon) * action_prob +
                       torch.ones_like(action_prob) * epsilon / action_num)
        action_prob[avail_actions == 0] = 0.0  # set avail action

        action_prob = action_prob / action_prob.sum(
            dim=-1, keepdim=True)  # in case action_prob.sum != 1
        action_prob[avail_actions == 0] = 0.0
        action_prob = action_prob.to(device)
        return action_prob

    def _cal_td_target(self, batch, q_targets):  # compute TD(lambda)
        """ input:
                batch: dict(o, s, u, r, u_onehot, avail_u, padded, isover, actor_inputs, critic_inputs)
                q_targets: Q value of target critic network, shape (ep_num, tr_num, n_agents)
            output:
                lambda_return: TD lambda return, shape (ep_num, tr_num, n_agents)
        """
        ep_num = batch['r'].shape[0]
        tr_num = batch['r'].shape[1]
        mask = (1 - batch['padded'].float()).repeat(1, 1,
                                                    self.n_agents).to(device)
        isover = (1 - batch['isover'].float()).repeat(1, 1, self.n_agents).to(
            device)  # used for setting last transition's q_target to 0
        # reshape reward: from (ep_num, tr_num, 1) to (ep_num, tr_num, n_agents)
        r = batch['r'].repeat((1, 1, self.n_agents)).to(device)
        # compute n_step_return
        n_step_return = torch.zeros((ep_num, tr_num, self.n_agents,
                                     tr_num)).to(device)
        for tr_id in range(tr_num - 1, -1, -1):
            n_step_return[:, tr_id, :, 0] = (
                r[:, tr_id] + self.gamma * q_targets[:, tr_id] *
                isover[:, tr_id]) * mask[:, tr_id]
            for n in range(1, tr_num - tr_id):
                n_step_return[:, tr_id, :, n] = (
                    r[:, tr_id] + self.gamma *
                    n_step_return[:, tr_id + 1, :, n - 1]) * mask[:, tr_id]

        lambda_return = torch.zeros((ep_num, tr_num, self.n_agents)).to(device)
        for tr_id in range(tr_num):
            returns = torch.zeros((ep_num, self.n_agents)).to(device)
            for n in range(1, tr_num - tr_id):
                returns += pow(self.td_lambda,
                               n - 1) * n_step_return[:, tr_id, :, n - 1]
            lambda_return[:, tr_id] = (1 - self.td_lambda) * returns + \
                                            pow(self.td_lambda, tr_num - tr_id - 1) * \
                                            n_step_return[:, tr_id, :, tr_num - tr_id - 1]
        return lambda_return

    def _critic_learn(self, batch):
        """ input:
                batch: dict(o, s, u, r, u_onehot, avail_u, padded, isover, actor_inputs, critic_inputs)
            output:
                q_values: Q value of eval critic network, shape (ep_num, tr_num, n_agents, n_actions)
        """
        u = batch['u']  # shape (ep_num, tr_num, agent, n_actions)
        u_next = torch.zeros_like(u, dtype=torch.long)
        u_next[:, :-1] = u[:, 1:]
        mask = (1 - batch['padded'].float()).repeat(1, 1,
                                                    self.n_agents).to(device)

        # get q value for every agent and every action, shape (ep_num, tr_num, n_agents, n_actions)
        q_evals, q_next_target = self._get_critic_output(batch)
        q_values = q_evals.clone()  # used for function return

        # get q valur for every agent
        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)
        q_next_target = torch.gather(
            q_next_target, dim=3, index=u_next).squeeze(3)

        targets = self._cal_td_target(batch, q_next_target)

        td_error = targets.detach() - q_evals
        masked_td_error = mask * td_error  # mask padded data

        loss = (masked_td_error**
                2).sum() / mask.sum()  # mask.sum: avail transition num

        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_parameters,
                                       self.grad_norm_clip)
        self.critic_optimizer.step()
        return q_values

    def _actor_learn(self, batch, epsilon, q_values):
        """ input:
                batch: dict(o, s, u, r, u_onehot, avail_u, padded, isover, actor_inputs, critic_inputs)
                epsilon (float): e-greedy discount
                q_values: Q value of eval critic network, shape (ep_num, tr_num, n_agents, n_actions)
        """
        action_prob = self._get_actor_output(batch, epsilon)  # prob of u

        # mask: used to compute TD-error, filling data should not affect learning
        u = batch['u']
        mask = (1 - batch['padded'].float()).repeat(1, 1, self.n_agents).to(
            device)  # shape (ep_num, tr_num, 3)

        q_taken = torch.gather(q_values, dim=3, index=u).squeeze(3)  # Q(u_a)
        pi_taken = torch.gather(
            action_prob, dim=3,
            index=u).squeeze(3)  # prob of act that agent a choosen
        pi_taken[mask == 0] = 1.0  # prevent log overflow
        log_pi_taken = torch.log(pi_taken)

        # advantage
        baseline = (q_values * action_prob).sum(
            dim=3, keepdim=True).squeeze(3).detach()
        advantage = (q_taken - baseline).detach()
        loss = -((advantage * log_pi_taken) * mask).sum() / mask.sum()
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_parameters,
                                       self.grad_norm_clip)
        self.actor_optimizer.step()

    def learn(self, batch, epsilon):
        """ input:
                batch: dict(o, s, u, r, u_onehot, avail_u, padded, isover, actor_inputs, critic_inputs)
                epsilon (float): e-greedy discount
        """
        ep_num = batch['r'].shape[0]
        self.train_rnn_h = self.init_hidden(ep_num)
        self.train_rnn_h = self.train_rnn_h.to(device)

        q_values = self._critic_learn(batch)
        self._actor_learn(batch, epsilon, q_values)

    def sync_target(self, decay=0):
        for param, target_param in zip(self.model.parameters(),
                                       self.target_model.parameters()):
            target_param.data.copy_((1 - decay) * param.data +
                                    decay * target_param.data)
