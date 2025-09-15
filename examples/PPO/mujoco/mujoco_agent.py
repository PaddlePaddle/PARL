#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import numpy as np


class MujocoAgent(parl.Agent):
    """ Agent of PPO env

    Args:
        algorithm (`parl.Algorithm`): algorithm to be used in this agent.
        config (dict): configs that used in this agent
    """
    def __init__(self, algorithm, config):
        super(MujocoAgent, self).__init__(algorithm)

        self.config = config
        self.kl_targ = self.config['kl_targ']
        # Adaptive kl penalty coefficient
        self.beta = 1.0  # dynamically adjusted D_KL loss multiplier
        self.lr_multiplier = 1.0  # dynamically adjust lr when D_KL out of control

        self.value_learn_buffer = None

    def sample(self, obs):
        """ Sample action from current policy given observation

        Args:
            obs (np.array): observation, shape([batch_size] + obs_shape)
        """
        obs = paddle.to_tensor(obs, dtype='float32')
        action = self.alg.sample(obs)
        action_numpy = action.detach().numpy()[0]
        return action_numpy

    def predict(self, obs):
        """ Predict action from current policy given observation

        Args:
            obs (np.array): observation, shape([batch_size] + obs_shape)
        """
        obs = paddle.to_tensor(obs, dtype='float32')
        action = self.alg.predict(obs)
        action_numpy = action.detach().numpy()[0]
        return action_numpy

    def value(self, obs):
        """ use the model to predict obs values

        Args:
            obs (torch tensor): observation, shape([batch_size] + obs_shape)
        """
        obs = paddle.to_tensor(obs, dtype='float32')
        value = self.alg.value(obs)
        value = value.detach().numpy()
        return value

    def _batch_policy_learn(self, obs, actions, advantages):
        obs = paddle.to_tensor(obs)
        actions = paddle.to_tensor(actions)
        advantages = paddle.to_tensor(advantages)

        loss, kl, entropy = self.alg.policy_learn(
            obs, actions, advantages, beta=self.beta, lr_multiplier=self.lr_multiplier)
        return loss, kl, entropy

    def _batch_value_learn(self, obs, discount_sum_rewards):
        obs = paddle.to_tensor(obs)
        discount_sum_rewards = paddle.to_tensor(discount_sum_rewards)
        loss = self.alg.value_learn(obs, discount_sum_rewards)
        return loss

    def policy_learn(self, obs, actions, advantages):
        """ policy learn
        """
        self.alg.sync_old_policy()

        all_loss, all_kl = [], []
        for _ in range(self.config['policy_learn_times']):
            loss, kl, entropy = self._batch_policy_learn(obs, actions, advantages)
            loss, kl, entropy = loss.numpy()[0], kl.numpy()[0], entropy.numpy()[0]
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
            all_loss.append(loss)
            all_kl.append(kl)

        if kl > self.kl_targ * 2:  # servo beta to reach D_KL target
            self.beta = np.minimum(35, 1.5 * self.beta)  # max clip beta
            if self.beta > 30 and self.lr_multiplier > 0.1:
                self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2:
            self.beta = np.maximum(1 / 35, self.beta / 1.5)  # min clip beta
            if self.beta < (1 / 30) and self.lr_multiplier < 10:
                self.lr_multiplier *= 1.5
        return loss, kl, self.beta, self.lr_multiplier, entropy

    def value_learn(self, obs, discount_sum_rewards):
        """ value learn
        """
        data_size = obs.shape[0]
        num_batches = max(data_size // self.config['value_batch_size'], 1)
        batch_size = data_size // num_batches

        if self.value_learn_buffer is None:
            obs_train, discount_sum_rewards_train = obs, discount_sum_rewards
        else:
            obs_train = np.concatenate([obs, self.value_learn_buffer[0]])
            discount_sum_rewards_train = np.concatenate([discount_sum_rewards, self.value_learn_buffer[1]])
        self.value_learn_buffer = (obs, discount_sum_rewards)

        all_loss = []
        y_hat = self.alg.model.value(paddle.to_tensor(obs)).numpy().reshape(
            [-1])  # check explained variance prior to update
        old_exp_var = 1 - np.var(discount_sum_rewards - y_hat) / np.var(discount_sum_rewards)

        for _ in range(self.config['value_learn_times']):
            random_ids = np.arange(obs_train.shape[0])
            np.random.shuffle(random_ids)
            shuffle_obs_train = obs_train[random_ids]
            shuffle_discount_sum_rewards_train = discount_sum_rewards_train[random_ids]
            start = 0
            while start < data_size:
                end = start + batch_size
                loss = self._batch_value_learn(shuffle_obs_train[start:end, :],
                                               shuffle_discount_sum_rewards_train[start:end])
                loss = loss.numpy()[0]
                all_loss.append(loss)
                start += batch_size
        y_hat = self.alg.model.value(paddle.to_tensor(obs)).numpy().reshape(
            [-1])  # check explained variance prior to update
        value_loss = np.mean(np.square(y_hat - discount_sum_rewards))  # explained variance after update
        exp_var = 1 - np.var(discount_sum_rewards - y_hat) / np.var(discount_sum_rewards)
        return value_loss, exp_var, old_exp_var
