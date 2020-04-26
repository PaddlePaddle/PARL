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

import warnings
warnings.simplefilter('default')

import numpy as np
from copy import deepcopy
from paddle import fluid
from parl.core.fluid import layers
from parl.core.fluid.algorithm import Algorithm

__all__ = ['PPO']


class PPO(Algorithm):
    def __init__(self,
                 model,
                 act_dim=None,
                 policy_lr=None,
                 value_lr=None,
                 epsilon=0.2):
        """ PPO algorithm
        
        Args:
            model (parl.Model): model defining forward network of policy and value.
            act_dim (float): dimension of the action space.
            policy_lr (float): learning rate of the policy model. 
            value_lr (float): learning rate of the value model.
            epsilon (float): epsilon used in the CLIP loss (default 0.2).
        """
        self.model = model
        # Used to calculate probability of action in old policy
        self.old_policy_model = deepcopy(model.policy_model)

        assert isinstance(act_dim, int)
        assert isinstance(policy_lr, float)
        assert isinstance(value_lr, float)
        assert isinstance(epsilon, float)
        self.act_dim = act_dim
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.epsilon = epsilon

    def _calc_logprob(self, actions, means, logvars):
        """ Calculate log probabilities of actions, when given means and logvars
            of normal distribution.
            The constant sqrt(2 * pi) is omitted, which will be eliminated in later.

        Args:
            actions: shape (batch_size, act_dim)
            means:   shape (batch_size, act_dim)
            logvars: shape (act_dim)

        Returns:
            logprob: shape (batch_size)
        """
        exp_item = layers.elementwise_div(
            layers.square(actions - means), layers.exp(logvars), axis=1)
        exp_item = -0.5 * layers.reduce_sum(exp_item, dim=1)

        vars_item = -0.5 * layers.reduce_sum(logvars)
        logprob = exp_item + vars_item
        return logprob

    def _calc_kl(self, means, logvars, old_means, old_logvars):
        """ Calculate KL divergence between old and new distributions
            See: https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback.E2.80.93Leibler_divergence

        Args:
            means: shape (batch_size, act_dim)
            logvars: shape (act_dim)
            old_means: shape (batch_size, act_dim)
            old_logvars: shape (act_dim)

        Returns:
            kl: shape (batch_size)
        """
        log_det_cov_old = layers.reduce_sum(old_logvars)
        log_det_cov_new = layers.reduce_sum(logvars)
        tr_old_new = layers.reduce_sum(layers.exp(old_logvars - logvars))
        kl = 0.5 * (layers.reduce_sum(
            layers.square(means - old_means) / layers.exp(logvars), dim=1) + (
                log_det_cov_new - log_det_cov_old) + tr_old_new - self.act_dim)
        return kl

    def predict(self, obs):
        """ Use the policy model of self.model to predict means and logvars of actions
        """
        means, logvars = self.model.policy(obs)
        return means

    def sample(self, obs):
        """ Use the policy model of self.model to sample actions
        """
        sampled_act = self.model.policy_sample(obs)
        return sampled_act

    def policy_learn(self, obs, actions, advantages, beta=None):
        """ Learn policy model with: 
                1. CLIP loss: Clipped Surrogate Objective 
                2. KLPEN loss: Adaptive KL Penalty Objective
            See: https://arxiv.org/pdf/1707.02286.pdf

        Args:
            obs: Tensor, (batch_size, obs_dim)
            actions: Tensor, (batch_size, act_dim)
            advantages: Tensor (batch_size, )
            beta: Tensor (1) or None
                  if None, use CLIP Loss; else, use KLPEN loss. 
        """
        old_means, old_logvars = self.old_policy_model.policy(obs)
        old_means.stop_gradient = True
        old_logvars.stop_gradient = True
        old_logprob = self._calc_logprob(actions, old_means, old_logvars)

        means, logvars = self.model.policy(obs)
        logprob = self._calc_logprob(actions, means, logvars)

        kl = self._calc_kl(means, logvars, old_means, old_logvars)
        kl = layers.reduce_mean(kl)

        if beta is None:  # Clipped Surrogate Objective
            pg_ratio = layers.exp(logprob - old_logprob)
            clipped_pg_ratio = layers.clip(pg_ratio, 1 - self.epsilon,
                                           1 + self.epsilon)
            surrogate_loss = layers.elementwise_min(
                advantages * pg_ratio, advantages * clipped_pg_ratio)
            loss = 0 - layers.reduce_mean(surrogate_loss)
        else:  # Adaptive KL Penalty Objective
            # policy gradient loss
            loss1 = 0 - layers.reduce_mean(
                advantages * layers.exp(logprob - old_logprob))
            # adaptive kl loss
            loss2 = kl * beta
            loss = loss1 + loss2
        optimizer = fluid.optimizer.AdamOptimizer(self.policy_lr)
        optimizer.minimize(loss)
        return loss, kl

    def value_predict(self, obs):
        """ Use value model of self.model to predict value of obs
        """
        return self.model.value(obs)

    def value_learn(self, obs, val):
        """ Learn the value model with square error cost
        """
        predict_val = self.model.value(obs)
        loss = layers.square_error_cost(predict_val, val)
        loss = layers.reduce_mean(loss)
        optimizer = fluid.optimizer.AdamOptimizer(self.value_lr)
        optimizer.minimize(loss)
        return loss

    def sync_old_policy(self):
        """ Synchronize weights of self.model.policy_model to self.old_policy_model
        """
        self.model.policy_model.sync_weights_to(self.old_policy_model)
