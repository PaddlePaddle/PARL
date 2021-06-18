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

import torch
import parl
from parl.utils.scheduler import PiecewiseScheduler, LinearDecayScheduler
import numpy as np

# torch use full CPU by default, which will decrease the performance. Use one thread for one actor here.
torch.set_num_threads(1)


class Agent(parl.Agent):
    def __init__(self, algorithm, config):
        super(Agent, self).__init__(algorithm)
        self.obs_shape = config['obs_shape']
        self.lr_scheduler = LinearDecayScheduler(config['start_lr'],
                                                 config['max_sample_steps'])

        self.entropy_coeff_scheduler = PiecewiseScheduler(
            config['entropy_coeff_scheduler'])

        self.device = torch.device("cuda" if torch.cuda.
                                   is_available() else "cpu")

    def sample(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        probs, values = self.alg.prob_and_value(obs)
        probs = probs.cpu().detach().numpy()
        values = values.cpu().detach().numpy()
        sample_actions = np.array(
            [np.random.choice(len(prob), 1, p=prob)[0] for prob in probs])
        return sample_actions, values

    def predict(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        predict_actions = self.alg.predict(obs)
        return predict_actions.cpu().detach().numpy()

    def value(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        values = self.alg.value(obs)
        return values

    def learn(self, obs, actions, advantages, target_values):
        obs = torch.FloatTensor(obs).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        target_values = torch.FloatTensor(target_values).to(self.device)

        lr = self.lr_scheduler.step(step_num=obs.shape[0])
        entropy_coeff = self.entropy_coeff_scheduler.step()
        total_loss, pi_loss, vf_losss, entropy = self.alg.learn(
            obs, actions, advantages, target_values, lr, entropy_coeff)

        return total_loss.cpu().detach().numpy(), pi_loss.cpu().detach().numpy(), \
            vf_losss.cpu().detach().numpy(), entropy.cpu().detach().numpy(), lr, entropy_coeff
