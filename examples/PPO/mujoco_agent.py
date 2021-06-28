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
import paddle


class MujocoAgent(parl.Agent):
    """ Agent of Mujoco env

    Args:
        algorithm (`parl.Algorithm`): algorithm to be used in this agent.
    """

    def __init__(self, algorithm):
        super(MujocoAgent, self).__init__(algorithm)

    def predict(self, obs):
        """ Predict action from current policy given observation

        Args:
            obs (np.array): observation
        """
        obs = paddle.to_tensor(obs, dtype='float32')
        action = self.alg.predict(obs)

        return action.detach().numpy()

    def sample(self, obs):
        """ Sample action from current policy given observation

        Args:
            obs (np.array): observation
        """
        obs = paddle.to_tensor(obs)
        value, action, action_log_probs = self.alg.sample(obs)

        return value.detach().numpy(), action.detach().numpy(), \
            action_log_probs.detach().numpy()

    def learn(self, next_value, gamma, gae_lambda, ppo_epoch, num_mini_batch,
              rollouts):
        """ Learn current batch of rollout for ppo_epoch epochs.

        Args:
            next_value (np.array): next predicted value for calculating advantage
            gamma (float): the discounting factor
            gae_lambda (float): lambda for calculating n step return
            ppo_epoch (int): number of epochs K
            num_mini_batch (int): number of mini-batches
            rollouts (RolloutStorage): the rollout storage that contains the current rollout
        """
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

                obs_batch = paddle.to_tensor(obs_batch)
                actions_batch = paddle.to_tensor(actions_batch)
                value_preds_batch = paddle.to_tensor(value_preds_batch)
                return_batch = paddle.to_tensor(return_batch)
                old_action_log_probs_batch = paddle.to_tensor(
                    old_action_log_probs_batch)
                adv_targ = paddle.to_tensor(adv_targ)

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
        """ Predict value from current value function given observation

        Args:
            obs (np.array): observation
        """
        obs = paddle.to_tensor(obs)
        val = self.alg.value(obs)

        return val.detach().numpy()
