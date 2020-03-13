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

import numpy as np

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RolloutStorage(object):
    def __init__(self, num_steps, obs_dim, act_dim):
        self.num_steps = num_steps
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.obs = np.zeros((num_steps + 1, obs_dim), dtype='float32')
        self.actions = np.zeros((num_steps, act_dim), dtype='float32')
        self.value_preds = np.zeros((num_steps + 1, ), dtype='float32')
        self.returns = np.zeros((num_steps + 1, ), dtype='float32')
        self.action_log_probs = np.zeros((num_steps, ), dtype='float32')
        self.rewards = np.zeros((num_steps, ), dtype='float32')

        self.masks = np.ones((num_steps + 1, ), dtype='bool')
        self.bad_masks = np.ones((num_steps + 1, ), dtype='bool')

        self.step = 0

    def append(self, obs, actions, action_log_probs, value_preds, rewards,
               masks, bad_masks):
        """
        print("obs")
        print(obs)
        print("masks")
        print(masks)
        print("rewards")
        print(rewards)
        exit()
        """
        self.obs[self.step + 1] = obs
        self.actions[self.step] = actions
        self.rewards[self.step] = rewards
        self.action_log_probs[self.step] = action_log_probs
        self.value_preds[self.step] = value_preds
        self.masks[self.step + 1] = masks
        self.bad_masks[self.step + 1] = bad_masks

        self.step = (self.step + 1) % self.num_steps

    def sample_batch(self,
                     next_value,
                     gamma,
                     gae_lambda,
                     num_mini_batch,
                     mini_batch_size=None):
        # calculate return and advantage first
        self.compute_returns(next_value, gamma, gae_lambda)
        advantages = self.returns[:-1] - self.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        # generate sample batch
        mini_batch_size = self.num_steps // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(self.num_steps)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:
            obs_batch = self.obs[:-1][indices]
            actions_batch = self.actions[indices]
            value_preds_batch = self.value_preds[:-1][indices]
            returns_batch = self.returns[:-1][indices]
            old_action_log_probs_batch = self.action_log_probs[indices]

            value_preds_batch = value_preds_batch.reshape(-1, 1)
            returns_batch = returns_batch.reshape(-1, 1)
            old_action_log_probs_batch = old_action_log_probs_batch.reshape(
                -1, 1)

            adv_targ = advantages[indices]
            adv_targ = adv_targ.reshape(-1, 1)

            yield obs_batch, actions_batch, value_preds_batch, returns_batch, old_action_log_probs_batch, adv_targ

    def after_update(self):
        self.obs[0] = np.copy(self.obs[-1])
        self.masks[0] = np.copy(self.masks[-1])
        self.bad_masks[0] = np.copy(self.bad_masks[-1])

    def compute_returns(self, next_value, gamma, gae_lambda):
        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.size)):
            delta = self.rewards[step] + gamma * self.value_preds[
                step + 1] * self.masks[step + 1] - self.value_preds[step]
            gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
            gae = gae * self.bad_masks[step + 1]
            self.returns[step] = gae + self.value_preds[step]
