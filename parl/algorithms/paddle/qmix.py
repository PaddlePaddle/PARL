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
import paddle.nn as nn
import parl
from copy import deepcopy
import paddle.nn.functional as F
from parl.utils.utils import check_model_method

__all__ = ['QMIX']


class QMIX(parl.Algorithm):
    def __init__(self,
                 agent_model,
                 qmixer_model,
                 double_q=True,
                 gamma=0.99,
                 lr=0.0005,
                 clip_grad_norm=None):
        """ QMIX algorithm

        Args:
            agent_model (parl.Model): agents' local q network for decision making.
            qmixer_model (parl.Model): A mixing network which takes local q values as input to construct a global Q network.
            double_q (bool): Double-DQN.
            gamma (float): discounted factor for reward computation.
            lr (float): learning rate.
            clip_grad_norm (None, or float): clipped value of gradients' global norm.
        """
        # checks
        check_model_method(agent_model, 'init_hidden', self.__class__.__name__)
        check_model_method(agent_model, 'forward', self.__class__.__name__)
        check_model_method(qmixer_model, 'forward', self.__class__.__name__)
        assert hasattr(qmixer_model, 'n_agents') and not callable(
            getattr(qmixer_model, 'n_agents',
                    None)), 'qmixer_model needs to have attribute n_agents'
        assert isinstance(gamma, float)
        assert isinstance(lr, float)

        self.agent_model = agent_model
        self.qmixer_model = qmixer_model
        self.target_agent_model = deepcopy(self.agent_model)
        self.target_qmixer_model = deepcopy(self.qmixer_model)

        self.n_agents = self.qmixer_model.n_agents
        self.double_q = double_q
        self.gamma = gamma
        self.lr = lr
        self.clip_grad_norm = clip_grad_norm

        self.params = list(self.agent_model.parameters())
        self.params += self.qmixer_model.parameters()
        if self.clip_grad_norm:
            clip = nn.ClipGradByGlobalNorm(clip_norm=self.clip_grad_norm)
            self.optimizer = paddle.optimizer.RMSProp(
                parameters=self.params,
                learning_rate=self.lr,
                rho=0.99,
                epsilon=1e-5,
                grad_clip=clip)
        else:
            self.optimizer = paddle.optimizer.RMSProp(
                parameters=self.params,
                learning_rate=self.lr,
                rho=0.99,
                epsilon=1e-5)

    def _init_hidden_states(self, batch_size):
        self.hidden_states = self.agent_model.init_hidden().unsqueeze(
            0).expand(shape=(batch_size, self.n_agents, -1))
        self.target_hidden_states = self.target_agent_model.init_hidden(
        ).unsqueeze(0).expand(shape=(batch_size, self.n_agents, -1))

    def predict_local_q(self, obs, hidden_state):
        return self.agent_model(obs, hidden_state)

    def learn(self, state_batch, actions_batch, reward_batch, terminated_batch,
              obs_batch, available_actions_batch, filled_batch):
        """
        Args:
            state_batch (paddle.Tensor):             (batch_size, T, state_shape)
            actions_batch (paddle.Tensor):           (batch_size, T, n_agents)
            reward_batch (paddle.Tensor):            (batch_size, T, 1)
            terminated_batch (paddle.Tensor):        (batch_size, T, 1)
            obs_batch (paddle.Tensor):               (batch_size, T, n_agents, obs_shape)
            available_actions_batch (paddle.Tensor): (batch_size, T, n_agents, n_actions)
            filled_batch (paddle.Tensor):            (batch_size, T, 1)
        Returns:
            loss (float): train loss
            td_error (float): train TD error
        """
        batch_size = state_batch.shape[0]
        episode_len = state_batch.shape[1]
        self._init_hidden_states(batch_size)
        n_actions = available_actions_batch.shape[-1]

        reward_batch = reward_batch[:, :-1, :]
        actions_batch = actions_batch[:, :-1, :]
        terminated_batch = terminated_batch[:, :-1, :]
        filled_batch = filled_batch[:, :-1, :]

        mask = (1 - filled_batch) * (1 - terminated_batch)

        local_qs = []
        target_local_qs = []
        for t in range(episode_len):
            obs = obs_batch[:, t, :, :]
            obs = obs.reshape(shape=(-1, obs_batch.shape[-1]))
            local_q, self.hidden_states = self.agent_model(
                obs, self.hidden_states)
            local_q = local_q.reshape(shape=(batch_size, self.n_agents, -1))
            local_qs.append(local_q)

            target_local_q, self.target_hidden_states = self.target_agent_model(
                obs, self.target_hidden_states)
            target_local_q = target_local_q.reshape(
                shape=(batch_size, self.n_agents, -1))
            target_local_qs.append(target_local_q)

        local_qs = paddle.stack(local_qs, axis=1)
        target_local_qs = paddle.stack(target_local_qs[1:], axis=1)

        actions_batch_one_hot = F.one_hot(actions_batch, num_classes=n_actions)
        chosen_action_local_qs = paddle.sum(
            local_qs[:, :-1, :, :] * actions_batch_one_hot, axis=-1)
        # mask unavailable actions
        target_unavailable_actions_mask = (
            available_actions_batch[:, 1:, :] == 0).cast('float32')
        target_local_qs -= 1e8 * target_unavailable_actions_mask
        if self.double_q:
            local_qs_detach = local_qs.clone().detach()
            unavailable_actions_mask = (
                available_actions_batch == 0).cast('float32')
            local_qs_detach -= 1e8 * unavailable_actions_mask
            cur_max_actions = paddle.argmax(
                local_qs_detach[:, 1:], axis=-1, keepdim=False)
            cur_max_actions_one_hot = F.one_hot(
                cur_max_actions, num_classes=n_actions)
            target_local_max_qs = paddle.sum(
                target_local_qs * cur_max_actions_one_hot, axis=-1)
        else:
            target_local_max_qs = target_local_qs.max(axis=3)

        chosen_action_global_qs = self.qmixer_model(chosen_action_local_qs,
                                                    state_batch[:, :-1, :])
        target_global_max_qs = self.target_qmixer_model(
            target_local_max_qs, state_batch[:, 1:, :])

        target = reward_batch + self.gamma * (
            1 - terminated_batch) * target_global_max_qs
        td_error = target.detach() - chosen_action_global_qs
        masked_td_error = td_error * mask
        mean_td_error = masked_td_error.sum() / mask.sum()
        loss = (masked_td_error**2).sum() / mask.sum()

        self.optimizer.clear_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss), float(mean_td_error)

    def sync_target(self):
        self.agent_model.sync_weights_to(self.target_agent_model)
        self.qmixer_model.sync_weights_to(self.target_qmixer_model)
