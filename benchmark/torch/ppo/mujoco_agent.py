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


class MujocoAgent(parl.Agent):
    def __init__(self, algorithm, device):
        super(MujocoAgent, self).__init__(algorithm)
        self.device = device

    def predict(self, obs):
        obs = torch.from_numpy(obs).float().to(self.device)
        action = self.alg.predict(obs)

        return action.cpu().detach().numpy()

    def sample(self, obs):
        obs = torch.from_numpy(obs).to(self.device)
        value, action, action_log_probs = self.alg.sample(obs)

        return value.cpu().detach().numpy(), action.cpu().detach().numpy(), \
            action_log_probs.cpu().detach().numpy()

    def learn(self, next_value, gamma, gae_lambda, ppo_epoch, num_mini_batch,
              rollouts):
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(ppo_epoch):
            data_generator = rollouts.sample_batch(next_value, gamma,
                                                   gae_lambda, num_mini_batch)

            for sample in data_generator:
                obs_batch, actions_batch, \
                    value_preds_batch, return_batch, old_action_log_probs_batch, \
                            adv_targ = sample

                obs_batch = torch.from_numpy(obs_batch).to('cuda')
                actions_batch = torch.from_numpy(actions_batch).to('cuda').to(
                    'cuda')
                value_preds_batch = torch.from_numpy(value_preds_batch).to(
                    'cuda')
                return_batch = torch.from_numpy(return_batch).to('cuda')
                old_action_log_probs_batch = torch.from_numpy(
                    old_action_log_probs_batch).to('cuda')
                adv_targ = torch.from_numpy(adv_targ).to('cuda')

                value_loss, action_loss, dist_entropy = self.alg.learn(
                    obs_batch, actions_batch, value_preds_batch, return_batch,
                    old_action_log_probs_batch, adv_targ)

                value_loss_epoch += value_loss
                action_loss_epoch += action_loss
                dist_entropy_epoch += dist_entropy

        num_updates = ppo_epoch * num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

    def value(self, obs):
        obs = torch.from_numpy(obs).to(self.device)
        val = self.alg.value(obs)

        return val.cpu().detach().numpy()
