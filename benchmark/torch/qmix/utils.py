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


class OneHotTransform(object):
    def __init__(self, out_dim):
        self.out_dim = out_dim

    def __call__(self, agent_id):
        assert agent_id < self.out_dim
        one_hot_id = np.zeros(self.out_dim, dtype='float32')
        one_hot_id[agent_id] = 1.0
        return one_hot_id


class EpsilonGreedy:
    def __init__(self,
                 action_nb,
                 agent_nb,
                 final_step,
                 epsilon_start=float(1),
                 epsilon_end=0.05):
        self.epsilon = epsilon_start
        self.initial_epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.action_nb = action_nb
        self.final_step = final_step
        self.agent_nb = agent_nb

    def act(self, value_action, avail_actions):
        if np.random.random() > self.epsilon:
            action = value_action.max(dim=1)[1].cpu().detach().numpy()
        else:
            action = torch.distributions.Categorical(
                avail_actions).sample().long().cpu().detach().numpy()
        return action

    def epislon_decay(self, step):
        progress = step / self.final_step

        decay = self.initial_epsilon - progress
        if decay <= self.epsilon_end:
            decay = self.epsilon_end
        self.epsilon = decay
