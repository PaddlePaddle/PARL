#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from parl.utils.scheduler import LinearDecayScheduler


class PPOAgent(parl.Agent):
    def __init__(self, algorithm, config):
        super(PPOAgent, self).__init__(algorithm)

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.
                                   is_available() else "cpu")
        if self.config['lr_decay']:
            self.lr_scheduler = LinearDecayScheduler(
                self.config['initial_lr'], self.config['num_updates'])

    def predict(self, obs):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action = self.alg.predict(obs)
        action_numpy = action.cpu().detach().numpy().flatten()
        return action_numpy

    def sample(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        value, action, action_log_probs, action_entropy = self.alg.sample(obs)

        value_numpy = value.cpu().detach().numpy()
        action_numpy = action.cpu().detach().numpy()
        action_log_probs_numpy = action_log_probs.cpu().detach().numpy()
        action_entropy_numpy = action_entropy.cpu().detach().numpy()
        return value_numpy, action_numpy, action_log_probs_numpy, action_entropy_numpy

    def value(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        value = self.alg.value(obs)
        value = value.cpu().detach().numpy()
        return value

    def learn(self, rollout):
        value_loss_epoch = 0
        action_loss_epoch = 0
        entropy_loss_epoch = 0
        if self.config['lr_decay']:
            lr = self.lr_scheduler.step(step_num=1)
        else:
            lr = None

        minibatch_size = int(
            self.config['batch_size'] // self.config['num_minibatches'])

        indexes = np.arange(self.config['batch_size'])
        for epoch in range(self.config['update_epochs']):
            np.random.shuffle(indexes)
            for start in range(0, self.config['batch_size'], minibatch_size):
                end = start + minibatch_size
                sample_idx = indexes[start:end]

                batch_obs, batch_action, batch_logprob, batch_adv, batch_return, batch_value = rollout.sample_batch(
                    sample_idx)

                batch_obs = torch.from_numpy(batch_obs).to(self.device)
                batch_action = torch.from_numpy(batch_action).to(self.device)
                batch_logprob = torch.from_numpy(batch_logprob).to(self.device)
                batch_adv = torch.from_numpy(batch_adv).to(self.device)
                batch_return = torch.from_numpy(batch_return).to(self.device)
                batch_value = torch.from_numpy(batch_value).to(self.device)

                value_loss, action_loss, entropy_loss = self.alg.learn(
                    batch_obs, batch_action, batch_value, batch_return,
                    batch_logprob, batch_adv, lr)

                value_loss_epoch += value_loss
                action_loss_epoch += action_loss
                entropy_loss_epoch += entropy_loss

        update_steps = self.config['update_epochs'] * self.config['batch_size']
        value_loss_epoch /= update_steps
        action_loss_epoch /= update_steps
        entropy_loss_epoch /= update_steps

        return value_loss_epoch, action_loss_epoch, entropy_loss_epoch, lr
