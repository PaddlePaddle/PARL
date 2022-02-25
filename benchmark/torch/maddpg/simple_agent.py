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
import torch
import numpy as np


class MAAgent(parl.Agent):
    def __init__(self,
                 algorithm,
                 agent_index=None,
                 obs_dim_n=None,
                 act_dim_n=None,
                 batch_size=None,
                 speedup=False):
        assert isinstance(agent_index, int)
        assert isinstance(obs_dim_n, list)
        assert isinstance(act_dim_n, list)
        assert isinstance(batch_size, int)
        assert isinstance(speedup, bool)
        self.agent_index = agent_index
        self.obs_dim_n = obs_dim_n
        self.act_dim_n = act_dim_n
        self.batch_size = batch_size
        self.speedup = speedup
        self.n = len(act_dim_n)

        self.min_memory_size = batch_size * 25  # batch_size * args.max_episode_len
        self.global_train_step = 0
        self.device = torch.device("cuda" if torch.cuda.
                                   is_available() else "cpu")

        super(MAAgent, self).__init__(algorithm)

        # Attention: In the beginning, sync target model totally.
        self.alg.sync_target(decay=0)

    def predict(self, obs, use_target_model=False):
        """ predict action by model or target_model
        """
        obs = torch.FloatTensor(obs.reshape(1, -1)).to(self.device)
        act = self.alg.predict(obs, use_target_model=use_target_model)
        act_numpy = act.detach().cpu().numpy().flatten()
        return act_numpy

    def learn(self, agents, rpms):
        """ sample batch, compute q_target and train
        """
        self.global_train_step += 1

        # only update parameter every 100 steps
        if self.global_train_step % 100 != 0:
            return 0.0
        
        rpm = rpms[self.agent_index]

        if rpm.size() <= self.min_memory_size:
            return 0.0

        batch_obs_n = []
        batch_act_n = []
        batch_obs_next_n = []

        # sample batch
        rpm_sample_index = rpm.make_index(self.batch_size)
        for i in range(self.n):
            batch_obs, batch_act, _, batch_obs_next, _ \
                = rpms[i].sample_batch_by_index(rpm_sample_index)
            batch_obs_n.append(batch_obs)
            batch_act_n.append(batch_act)
            batch_obs_next_n.append(batch_obs_next)
        _, _, batch_rew, _, batch_isOver = rpm.sample_batch_by_index(
            rpm_sample_index)
        batch_obs_n = [
            torch.FloatTensor(obs).to(self.device) for obs in batch_obs_n
        ]
        batch_act_n = [
            torch.FloatTensor(act).to(self.device) for act in batch_act_n
        ]
        batch_rew = torch.FloatTensor(batch_rew).to(self.device)
        batch_isOver = torch.FloatTensor(batch_isOver).to(self.device)

        # compute target q
        target_act_next_n = []
        batch_obs_next_n = [
            torch.FloatTensor(obs).to(self.device) for obs in batch_obs_next_n
        ]
        for i in range(self.n):
            target_act_next = agents[i].alg.predict(
                batch_obs_next_n[i], use_target_model=True)
            target_act_next = target_act_next.detach()
            target_act_next_n.append(target_act_next)
        target_q_next = self.alg.Q(
            batch_obs_next_n, target_act_next_n, use_target_model=True)
        target_q = batch_rew + self.alg.gamma * (
            1.0 - batch_isOver) * target_q_next.detach()

        # learn
        critic_cost = self.alg.learn(batch_obs_n, batch_act_n, target_q)
        critic_cost = critic_cost.cpu().detach().numpy()

        return critic_cost
