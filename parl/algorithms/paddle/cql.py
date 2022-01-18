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
from paddle.distribution import Normal
import paddle.nn.functional as F
from parl.utils.utils import check_model_method
from copy import deepcopy
import numpy as np

__all__ = ['CQL']


class CQL(parl.Algorithm):
    def __init__(self,
                 model,
                 gamma=None,
                 tau=None,
                 actor_lr=None,
                 critic_lr=None,
                 policy_eval_start=40000,
                 with_automatic_entropy_tuning=True,
                 with_lagrange=False,
                 lagrange_thresh=10.0,
                 min_q_version=3,
                 min_q_weight=5.0,
                 alpha=1.0):
        """ CQL algorithm
        Args:
            model (parl.Model): forward network of actor and critic.
            gamma (float): discounted factor for reward computation
            tau (float): decay coefficient when updating the weights of self.target_model with self.model
            actor_lr (float): learning rate of the actor model
            critic_lr (float): learning rate of the critic model
            policy_eval_start (int): try doing behaivoral cloning at the beginning, 40000 or 10000 work similarly
            with_automatic_entropy_tuning (bool): train with automatic entropy tuning in Actor.
            with_lagrange (bool): train with lagrange
            lagrange_thresh (float): the value of tau, corresponds to the CQL(lagrange) version, suggest 10.0 in mujoco and 5.0 in Franka kitchen or Adroit domains
            min_q_version (int): min_q_version = 3 (CQL(H)), = 2 (CQL(rho)), will be set to <0 in cql if not using lagrange
            min_q_weight (float): the value of alpha in Critic loss, suggest 5.0 or 10.0 if not using lagrange
            alpha (float): the value of alpha(temperature parameter) in Actor loss, determines the relative importance of entropy term against the reward
        """
        # check model method
        check_model_method(model, 'value', self.__class__.__name__)
        check_model_method(model, 'policy', self.__class__.__name__)
        check_model_method(model, 'get_actor_params', self.__class__.__name__)
        check_model_method(model, 'get_critic_params', self.__class__.__name__)
        assert isinstance(gamma, float)
        assert isinstance(tau, float)
        assert isinstance(actor_lr, float)
        assert isinstance(critic_lr, float)
        assert isinstance(with_automatic_entropy_tuning, bool)
        assert isinstance(with_lagrange, bool)
        assert isinstance(lagrange_thresh, float)
        assert isinstance(min_q_version, int)
        assert isinstance(min_q_weight, float)
        assert isinstance(alpha, float)

        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_prime = 0.0
        self.policy_eval_start = policy_eval_start
        self.with_automatic_entropy_tuning = with_automatic_entropy_tuning
        self.with_lagrange = with_lagrange
        self.lagrange_thresh = lagrange_thresh
        self.min_q_version = -1 if self.with_lagrange else min_q_version
        self.min_q_weight = min_q_weight
        self.alpha = alpha
        self.temp = 1.0
        self.num_random = 10
        self._current_steps = 0
        self.model = model
        self.target_model = deepcopy(self.model)
        self.actor_optimizer = paddle.optimizer.Adam(
            learning_rate=actor_lr, parameters=self.model.get_actor_params())
        self.critic_optimizer = paddle.optimizer.Adam(
            learning_rate=critic_lr, parameters=self.model.get_critic_params())

        if self.with_automatic_entropy_tuning:
            self.target_entropy = None
            self.log_alpha = paddle.zeros([1])
            self.log_alpha.stop_gradient = False
            self.alpha_optimizer = paddle.optimizer.Adam(
                learning_rate=actor_lr, parameters=[self.log_alpha])
        if self.with_lagrange:
            self.target_action_gap = self.lagrange_thresh
            self.log_alpha_prime = paddle.zeros([1])
            self.log_alpha_prime.stop_gradient = False
            self.alpha_prime_optimizer = paddle.optimizer.Adam(
                learning_rate=critic_lr, parameters=[self.log_alpha_prime])

    def predict(self, obs):
        """ Define the predicting process, e.g,. use the policy model to predict actions.
        """
        act_mean, _ = self.model.policy(obs)
        action = paddle.tanh(act_mean)
        return action

    def sample(self, obs):
        """ Define the sampling process. This function returns an action with noise to perform exploration.
        """
        act_mean, act_log_std = self.model.policy(obs)
        normal = Normal(act_mean, act_log_std.exp())
        # for reparameterization trick  (mean + std * N(0, 1))
        x_t = normal.sample([1])
        action = paddle.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= paddle.log((1 - action.pow(2)) + 1e-6)
        log_prob = paddle.sum(log_prob, axis=-1, keepdim=True)
        return action[0], log_prob[0]

    def learn(self, obs, action, reward, next_obs, terminal):
        """ Define the loss function and create an optimizer to minize the loss.
        """
        self._current_steps += 1
        critic_loss, mse = self._critic_learn(obs, action, reward, next_obs,
                                              terminal)
        actor_loss, min_q = self._actor_learn(obs, action)

        self.sync_target()
        return critic_loss, mse, actor_loss, min_q

    def _critic_learn(self, obs, action, reward, next_obs, terminal):
        with paddle.no_grad():
            next_action, next_log_pro = self.sample(next_obs)
            q1_next, q2_next = self.target_model.value(next_obs, next_action)
            target_Q = paddle.minimum(q1_next,
                                      q2_next)  # a little different from SAC
            target_Q = reward + self.gamma * (1. - terminal) * target_Q
        cur_q1, cur_q2 = self.model.value(obs, action)

        qf1_loss = F.mse_loss(cur_q1, target_Q)
        qf2_loss = F.mse_loss(cur_q2, target_Q)
        mse = qf2_loss + qf1_loss
        ## add CQL
        random_actions_tensor = paddle.uniform(
            shape=[cur_q2.shape[0] * self.num_random, action.shape[-1]])
        curr_actions_tensor, curr_log_pis = self._get_policy_actions(obs)
        new_curr_actions_tensor, new_log_pis = self._get_policy_actions(
            next_obs)

        q1_rand, q2_rand = self._get_tensor_values(obs, random_actions_tensor)
        q1_curr_actions, q2_curr_actions = self._get_tensor_values(
            obs, curr_actions_tensor)
        q1_next_actions, q2_next_actions = self._get_tensor_values(
            obs, new_curr_actions_tensor)

        cat_q1 = paddle.concat(
            [q1_rand, cur_q1.unsqueeze(1), q1_curr_actions], 1)
        cat_q2 = paddle.concat(
            [q2_rand, cur_q2.unsqueeze(1), q2_curr_actions], 1)

        if self.min_q_version == 3:
            # importance sampled version
            random_density = np.log(0.5**curr_actions_tensor.shape[-1])
            cat_q1 = paddle.concat([
                q1_rand - random_density,
                q1_next_actions - new_log_pis.detach(),
                q1_curr_actions - curr_log_pis.detach()
            ], 1)
            cat_q2 = paddle.concat([
                q2_rand - random_density,
                q2_next_actions - new_log_pis.detach(),
                q2_curr_actions - curr_log_pis.detach()
            ], 1)

        min_qf1_loss = paddle.logsumexp(
            cat_q1 / self.temp,
            axis=1,
        ).mean() * self.min_q_weight * self.temp
        min_qf2_loss = paddle.logsumexp(
            cat_q2 / self.temp,
            axis=1,
        ).mean() * self.min_q_weight * self.temp

        # Subtract the log likelihood of data
        min_qf1_loss -= cur_q1.mean() * self.min_q_weight
        min_qf2_loss -= cur_q2.mean() * self.min_q_weight

        if self.with_lagrange:
            self.alpha_prime = paddle.clip(
                self.log_alpha_prime.exp(), min=0.0, max=1000000.0)
            min_qf1_loss = self.alpha_prime * (
                min_qf1_loss - self.target_action_gap)
            min_qf2_loss = self.alpha_prime * (
                min_qf2_loss - self.target_action_gap)

            self.alpha_prime_optimizer.clear_grad()
            alpha_prime_loss = (-min_qf1_loss - min_qf2_loss) * 0.5
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()

        qf1_loss = qf1_loss + min_qf1_loss
        qf2_loss = qf2_loss + min_qf2_loss
        ## CQL done

        critic_loss = qf1_loss + qf2_loss
        self.critic_optimizer.clear_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()
        return critic_loss, mse

    def _actor_learn(self, obs, action):
        act, log_pi = self.sample(obs)
        q1_pi, q2_pi = self.model.value(obs, act)
        min_q_pi = paddle.minimum(q1_pi, q2_pi)

        if self.with_automatic_entropy_tuning:
            if self.target_entropy is None:
                self.target_entropy = -np.prod(act.shape[-1]).item()
            alpha_loss = -(self.log_alpha *
                           (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.clear_grad()
            alpha_loss.backward(retain_graph=True)
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0

        actor_loss = (self.alpha * log_pi - min_q_pi).mean()

        if self._current_steps < self.policy_eval_start:
            """
            For the initial few epochs, try doing behaivoral cloning, if needed
            conventionally, there's not much difference in performance with having 20k 
            gradient steps here, or not having it
            """
            policy_log_prob = self._sample_log_prob(obs, action)
            actor_loss = (self.alpha * log_pi - policy_log_prob).mean()

        self.actor_optimizer.clear_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()
        return actor_loss, min_q_pi.mean()

    def _sample_log_prob(self, obs, action):
        def atanh(x):
            one_plus_x = (1 + x).clip(min=1e-6)
            one_minus_x = (1 - x).clip(min=1e-6)
            return 0.5 * paddle.log(one_plus_x / one_minus_x)

        raw_action = atanh(action)
        act_mean, act_log_std = self.model.policy(obs)

        normal = Normal(act_mean, act_log_std.exp())
        log_prob = normal.log_prob(raw_action)
        log_prob -= paddle.log(1 - action.pow(2) + 1e-6)
        return log_prob.sum(-1)

    def _get_policy_actions(self, obs):
        obs_temp = obs.unsqueeze(1).tile([1, self.num_random, 1]).reshape(
            [obs.shape[0] * self.num_random, obs.shape[1]])
        new_obs_actions, new_obs_log_pi = self.sample(obs_temp)
        return new_obs_actions, new_obs_log_pi.reshape(
            [obs.shape[0], self.num_random, 1])

    def _get_tensor_values(self, obs, actions):
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int(action_shape / obs_shape)
        obs_temp = obs.unsqueeze(1).tile([1, num_repeat, 1]).reshape(
            [obs.shape[0] * num_repeat, obs.shape[1]])
        q1, q2 = self.model.value(obs_temp, actions)
        q1 = q1.reshape([obs.shape[0], num_repeat, 1])
        q2 = q2.reshape([obs.shape[0], num_repeat, 1])
        return q1, q2

    def sync_target(self, decay=None):
        """ update the target network with the training network
        Args:
            decay(float): the decaying factor while updating the target network with the training network.
                        0 represents the **assignment**. None represents updating the target network slowly that depends on the hyperparameter `tau`.
        """
        if decay is None:
            decay = 1.0 - self.tau
        self.model.sync_weights_to(self.target_model, decay=decay)
