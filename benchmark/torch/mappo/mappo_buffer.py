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

# modified from https://github.com/marlbenchmark/on-policy

import torch
import numpy as np


class SeparatedReplayBuffer(object):
    def __init__(self, episode_length, env_num, gamma, gae_lambda, obs_shape,
                 share_obs_shape, act_space, use_popart):
        """  ReplayBuffer for each agent

        Args:
            model (parl.Model): model that contains both value network and policy network
            episode_length (int): max length for any episode
            env_num (int): Number of parallel envs to train
            gamma (float): discount factor for rewards
            gae_lambda (float): gae lambda parameter
            obs_shape (int): obs dim for single agent
            share_obs_shape (int): concatenated obs dim for all agents
            act_space (MultiDiscrete/Discrete): action space for single agent
            use_popart (bool): whether to use PopArt to normalize rewards
        """
        self.episode_length = episode_length
        self.env_num = env_num
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self._use_popart = use_popart

        self.share_obs = np.zeros(
            (self.episode_length + 1, self.env_num, share_obs_shape),
            dtype=np.float32)
        self.obs = np.zeros((self.episode_length + 1, self.env_num, obs_shape),
                            dtype=np.float32)
        self.value_preds = np.zeros((self.episode_length + 1, self.env_num, 1),
                                    dtype=np.float32)
        self.returns = np.zeros((self.episode_length + 1, self.env_num, 1),
                                dtype=np.float32)

        if act_space.__class__.__name__ == 'Discrete':
            act_shape = 1
        else:
            act_shape = act_space.shape

        self.actions = np.zeros((self.episode_length, self.env_num, act_shape),
                                dtype=np.float32)
        self.action_log_probs = np.zeros(
            (self.episode_length, self.env_num, act_shape), dtype=np.float32)
        self.rewards = np.zeros((self.episode_length, self.env_num, 1),
                                dtype=np.float32)
        self.masks = np.ones((self.episode_length + 1, self.env_num, 1),
                             dtype=np.float32)
        self.active_masks = np.ones_like(self.masks)

        self.step = 0

    def insert(self,
               share_obs,
               obs,
               actions,
               action_log_probs,
               value_preds,
               rewards,
               masks,
               active_masks=None):
        """ insert sample data into buffer
        """
        self.share_obs[self.step + 1] = share_obs.copy()
        self.obs[self.step + 1] = obs.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        """ update buffer after learn
        """
        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()

    def compute_returns(self, next_value, value_normalizer=None):
        """ compute return for each step
        """
        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.shape[0])):
            if self._use_popart:
                delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(self.value_preds[step + 1]) * \
                        self.masks[step + 1] - value_normalizer.denormalize(self.value_preds[step])
                gae = delta + self.gamma * self.gae_lambda * self.masks[
                    step + 1] * gae
                self.returns[step] = gae + value_normalizer.denormalize(
                    self.value_preds[step])
            else:
                delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - \
                        self.value_preds[step]
                gae = delta + self.gamma * self.gae_lambda * self.masks[
                    step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]

    def sample_batch(self, advantages, num_mini_batch=None):
        """sample data from replay memory for training
        """
        episode_length, env_num = self.rewards.shape[0:2]
        batch_size = env_num * episode_length
        mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [
            rand[i * mini_batch_size:(i + 1) * mini_batch_size]
            for i in range(num_mini_batch)
        ]

        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[2:])
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(
            -1, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, 1)

        for indices in sampler:
            share_obs_batch = share_obs[indices]
            obs_batch = obs[indices]
            actions_batch = actions[indices]
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]

            yield share_obs_batch, obs_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ
