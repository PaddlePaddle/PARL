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
import paddle
import paddle.nn.functional as F
from paddle.distribution import Categorical

from parl.algorithms.paddle.impala import vtrace

__all__ = ['IMPALA']


class VTraceLoss(object):
    def __init__(self,
                 behaviour_actions_log_probs,
                 target_actions_log_probs,
                 policy_entropy,
                 dones,
                 discount,
                 rewards,
                 values,
                 bootstrap_value,
                 entropy_coeff=-0.01,
                 vf_loss_coeff=0.5,
                 clip_rho_threshold=1.0,
                 clip_pg_rho_threshold=1.0):
        """Policy gradient loss with vtrace importance weighting.

        VTraceLoss takes tensors of shape [T, B, ...], where `B` is the
        batch_size. The reason we need to know `B` is for V-trace to properly
        handle episode cut boundaries.

        Args:
            behaviour_actions_log_probs: A float32 tensor of shape [T, B].
            target_actions_log_probs: A float32 tensor of shape [T, B].
            policy_entropy: A float32 tensor of shape [T, B].
            dones: A float32 tensor of shape [T, B].
            discount: A float32 scalar.
            rewards: A float32 tensor of shape [T, B].
            values: A float32 tensor of shape [T, B].
            bootstrap_value: A float32 tensor of shape [B].
        """

        self.vtrace_returns = vtrace.from_importance_weights(
            behaviour_actions_log_probs=behaviour_actions_log_probs,
            target_actions_log_probs=target_actions_log_probs,
            discounts=(~dones).astype('float32') * discount,
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_value,
            clip_rho_threshold=clip_rho_threshold,
            clip_pg_rho_threshold=clip_pg_rho_threshold)

        # The policy gradients loss
        self.pi_loss = -1.0 * paddle.sum(
            target_actions_log_probs * self.vtrace_returns.pg_advantages)

        # The baseline loss
        delta = values - self.vtrace_returns.vs
        self.vf_loss = 0.5 * paddle.sum(paddle.square(delta))

        # The entropy loss (We want to maximize entropy, so entropy_ceoff < 0)
        self.entropy = paddle.sum(policy_entropy)

        # The summed weighted loss
        self.total_loss = (self.pi_loss + self.vf_loss * vf_loss_coeff +
                           self.entropy * entropy_coeff)


class IMPALA(parl.Algorithm):
    def __init__(self,
                 model,
                 sample_batch_steps=None,
                 gamma=None,
                 vf_loss_coeff=None,
                 clip_rho_threshold=None,
                 clip_pg_rho_threshold=None):
        r""" IMPALA algorithm
        
        Args:
            model (parl.Model): forward network of policy and value
            sample_batch_steps (int): steps of each environment sampling.
            gamma (float): discounted factor for reward computation.
            vf_loss_coeff (float): coefficient of the value function loss.
            clip_rho_threshold (float): clipping threshold for importance weights (rho).
            clip_pg_rho_threshold (float): clipping threshold on rho_s in \rho_s \delta log \pi(a|x) (r + \gamma v_{s+1} - V(x_s)).
        """
        assert isinstance(sample_batch_steps, int)
        assert isinstance(gamma, float)
        assert isinstance(vf_loss_coeff, float)
        assert isinstance(clip_rho_threshold, float)
        assert isinstance(clip_pg_rho_threshold, float)
        self.sample_batch_steps = sample_batch_steps
        self.gamma = gamma
        self.vf_loss_coeff = vf_loss_coeff
        self.clip_rho_threshold = clip_rho_threshold
        self.clip_pg_rho_threshold = clip_pg_rho_threshold

        self.model = model

        clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=40.0)
        self.optimizer = paddle.optimizer.Adam(
            learning_rate=0.001,
            parameters=self.model.parameters(),
            grad_clip=clip)

    def _log_prob(self, logits, actions):
        """
        Args:
            logits: A float32 tensor of shape [B, NUM_ACTIONS]
            actions: An int64 tensor of shape [B].

        Returns:
            actions_log_probs: A
        """
        act_dim = logits.shape[-1]
        actions_onehot = F.one_hot(actions, act_dim)
        actions_log_probs = paddle.sum(
            F.log_softmax(logits) * actions_onehot, axis=-1)
        return actions_log_probs

    def learn(self, obs, actions, behaviour_logits, rewards, dones,
              learning_rate, entropy_coeff):
        """
        Args:
            obs: An float32 tensor of shape ([B] + observation_space).
                 E.g. [B, C, H, W] in atari.
            actions: An int64 tensor of shape [B].
            behaviour_logits: A float32 tensor of shape [B, NUM_ACTIONS].
            rewards: A float32 tensor of shape [B].
            dones: A float32 tensor of shape [B].
            learning_rate: float scalar of learning rate.
            entropy_coeff: float scalar of entropy coefficient.
        """

        values = self.model.value(obs)
        target_logits = self.model.policy(obs)

        target_policy_distribution = Categorical(target_logits)
        behaviour_policy_distribution = Categorical(behaviour_logits)

        policy_entropy = target_policy_distribution.entropy()

        target_actions_log_probs = self._log_prob(target_logits, actions)
        behaviour_actions_log_probs = self._log_prob(behaviour_logits, actions)

        # Calculating kl for debug
        kl = target_policy_distribution.kl_divergence(
            behaviour_policy_distribution)
        kl = paddle.mean(kl)
        """
        Split the tensor into batches at known episode cut boundaries. 
        [B * T] -> [T, B]
        """
        T = self.sample_batch_steps
        B = obs.shape[0] // T

        def split_batches(tensor):
            splited_tensor = paddle.reshape(tensor,
                                            [B, T] + list(tensor.shape[1:]))
            # transpose B and T
            return paddle.transpose(
                splited_tensor, [1, 0] + list(range(2, 1 + len(tensor.shape))))

        behaviour_actions_log_probs = split_batches(
            behaviour_actions_log_probs)
        target_actions_log_probs = split_batches(target_actions_log_probs)
        policy_entropy = split_batches(policy_entropy)
        dones = split_batches(dones)
        rewards = split_batches(rewards)
        values = split_batches(values)

        # [T, B] -> [T - 1, B] for V-trace calc.
        behaviour_actions_log_probs = behaviour_actions_log_probs[:-1]
        target_actions_log_probs = target_actions_log_probs[:-1]
        policy_entropy = policy_entropy[:-1]
        dones = dones[:-1]
        rewards = rewards[:-1]
        bootstrap_value = values[-1:]
        values = values[:-1]

        bootstrap_value = paddle.squeeze(bootstrap_value, 0)

        vtrace_loss = VTraceLoss(
            behaviour_actions_log_probs=behaviour_actions_log_probs,
            target_actions_log_probs=target_actions_log_probs,
            policy_entropy=policy_entropy,
            dones=dones,
            discount=self.gamma,
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_value,
            entropy_coeff=entropy_coeff,
            vf_loss_coeff=self.vf_loss_coeff,
            clip_rho_threshold=self.clip_rho_threshold,
            clip_pg_rho_threshold=self.clip_pg_rho_threshold)

        self.optimizer.set_lr(learning_rate)
        vtrace_loss.total_loss.backward()
        self.optimizer.step()
        self.optimizer.clear_grad()

        return vtrace_loss, kl

    def sample(self, obs):
        """
        Args:
            obs: An float32 tensor of shape ([B] + observation_space).
                 E.g. [B, C, H, W] in atari.
        """
        logits = self.model.policy(obs)
        policy_dist = Categorical(logits)

        probs = F.softmax(logits)
        return probs, logits
