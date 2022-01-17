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

# modified from https://github.com/marlbenchmark/on-policy

import torch
import numpy as np
import parl


class MAPPOgent(parl.Agent):
    def __init__(self, ppo_epoch, num_mini_batch, env_num, algorithm,
                 use_popart):
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self._use_popart = use_popart
        self.env_num = env_num
        self.value_normalizer = algorithm.value_normalizer
        self.device = torch.device("cuda" if torch.cuda.
                                   is_available() else "cpu")
        super(MAPPOgent, self).__init__(algorithm)

    @torch.no_grad()
    def sample(self, share_obs, obs):
        self.alg.model.actor.eval()
        self.alg.model.critic.eval()
        value, action, action_log_prob = self.alg.sample(share_obs, obs)
        return value.detach().cpu().numpy(), action.detach().cpu().numpy(
        ), action_log_prob.detach().cpu().numpy()

    def learn(self, buffer):
        self.alg.model.actor.train()
        self.alg.model.critic.train()
        if self._use_popart:
            advantages = buffer.returns[:
                                        -1] - self.value_normalizer.denormalize(
                                            buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]

        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}
        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0

        for _ in range(self.ppo_epoch):
            data_generator = buffer.feed_forward_generator(
                advantages, self.num_mini_batch)
            for sample in data_generator:
                value_loss, policy_loss, dist_entropy = self.alg.learn(sample)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch
        for k in train_info.keys():
            train_info[k] /= num_updates
        return train_info

    def value(self, share_obs):
        self.alg.model.actor.eval()
        self.alg.model.critic.eval()
        next_values = self.alg.value(share_obs).detach().cpu().numpy()
        next_values = np.array(np.split(next_values, self.env_num))
        return next_values
