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
from copy import deepcopy

__all__ = ['PPO_Mujoco']


class PPO_Mujoco(parl.Algorithm):
    def __init__(self, model, act_dim=None, loss_type='KLPEN', kl_targ=0.003, eta=50, clip_param=0.2, eps=1e-5):
        """ PPO algorithm for Mujoco
        
        Args:
            model (parl.Model): model defining forward network of policy and value.
            act_dim (float): dimension of the action space.
            loss_type (string): loss type of PPO algorithm, 'CLIP' or 'KLPEN'".
            kl_targ (float): D_KL target value.
            eta (float):  multiplier for D_KL-kl_targ hinge-squared loss.
            clip_param (float): epsilon used in the CLIP loss.
            eps (float): A small float value for numerical stability.
        """
        assert isinstance(act_dim, int)
        assert isinstance(clip_param, float)
        assert loss_type == 'CLIP' or loss_type == 'KLPEN'
        self.loss_type = loss_type
        self.act_dim = act_dim
        self.clip_param = clip_param
        self.eta = eta
        self.kl_targ = kl_targ

        self.model = model
        # Used to calculate probability of action in old policy
        self.old_policy_model = deepcopy(model.policy_model)

        self.policy_lr = self.model.policy_lr
        self.value_lr = self.model.value_lr
        self.policy_optimizer = paddle.optimizer.Adam(
            parameters=self.model.policy_model.parameters(), learning_rate=self.policy_lr, epsilon=eps)
        self.value_optimizer = paddle.optimizer.Adam(
            parameters=self.model.value_model.parameters(), learning_rate=self.value_lr, epsilon=eps)

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
        logp = -0.5 * paddle.sum(logvars)
        logp += -0.5 * paddle.sum((paddle.square(actions - means) / paddle.exp(logvars)), axis=1)
        logprob = logp
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
            entropy
        """
        log_det_cov_old = paddle.sum(old_logvars)
        log_det_cov_new = paddle.sum(logvars)
        tr_old_new = paddle.sum(paddle.exp(old_logvars - logvars))
        kl = 0.5 * paddle.mean(
            paddle.sum(paddle.square(means - old_means) / paddle.exp(logvars), axis=1) +
            (log_det_cov_new - log_det_cov_old) + tr_old_new - self.act_dim)

        entropy = 0.5 * (self.act_dim * (np.log(2 * np.pi) + 1) + paddle.sum(logvars))

        return kl, entropy

    def value(self, obs):
        """ Use value model of self.model to predict value of obs
        """
        return self.model.value(obs)

    def predict(self, obs):
        """ Use the policy model of self.model to predict means and logvars of actions
        """
        means, logvars = self.model.policy(obs)
        return means

    def sample(self, obs):
        """ Use the policy model of self.model to sample actions
        """
        means, logvars = self.model.policy(obs)
        sampled_act = means + (
            paddle.exp(logvars / 2.0) *  # stddev
            paddle.standard_normal(shape=(self.act_dim, ), dtype='float32'))
        return sampled_act

    def policy_learn(self, batch_obs, batch_action, batch_adv, beta, lr_multiplier):
        """ Learn policy model with: 
                1. CLIP loss: Clipped Surrogate Objective 
                2. KLPEN loss: Adaptive KL Penalty Objective
            See: https://arxiv.org/pdf/1707.02286.pdf

        Args:
            batch_obs: Tensor, (batch_size, obs_dim)
            batch_action: Tensor, (batch_size, act_dim)
            batch_adv: Tensor (batch_size, )
            beta: Tensor (1) or None. If None, use CLIP Loss; else, use KLPEN loss. 
            lr_multiplier: Tensor (1)
        """
        old_means, old_logvars = self.old_policy_model.policy(batch_obs)
        old_means.stop_gradient = True
        old_logvars.stop_gradient = True

        old_logprob = self._calc_logprob(batch_action, old_means, old_logvars)
        old_logprob.stop_gradient = True

        means, logvars = self.model.policy(batch_obs)
        logprob = self._calc_logprob(batch_action, means, logvars)
        kl, entropy = self._calc_kl(means, logvars, old_means, old_logvars)

        if self.loss_type == "KLPEN":
            loss1 = -(batch_adv * paddle.exp(logprob - old_logprob)).mean()
            loss2 = (kl * beta).mean()
            loss3 = self.eta * paddle.square(paddle.maximum(paddle.to_tensor(0.0), kl - 2.0 * self.kl_targ))
            loss = loss1 + loss2 + loss3
        elif self.loss_type == "CLIP":
            ratio = paddle.exp(logprob - old_logprob)
            surr1 = ratio * batch_adv
            surr2 = paddle.clip(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_adv
            loss = -paddle.minimum(surr1, surr2).mean()
        else:
            raise ValueError("Policy loss type error, 'CLIP' or 'KLPEN'")

        self.policy_optimizer.set_lr(self.policy_lr * lr_multiplier)

        self.policy_optimizer.clear_grad()
        loss.backward()
        self.policy_optimizer.step()
        return loss, kl, entropy

    def value_learn(self, batch_obs, batch_return):
        """ Learn the value model with square error cost
        """
        predict_val = self.model.value(batch_obs)
        predict_val = predict_val.reshape([-1])

        loss = (predict_val - batch_return).pow(2).mean()

        self.value_optimizer.clear_grad()
        loss.backward()
        self.value_optimizer.step()
        return loss

    def sync_old_policy(self):
        """ Synchronize weights of self.model.policy_model to self.old_policy_model
        """
        self.model.policy_model.sync_weights_to(self.old_policy_model)
