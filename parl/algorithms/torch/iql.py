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

import copy
import torch
import torch.nn as nn
import parl
import torch.nn.functional as F

EXP_ADV_MAX = 100.


def asymmetric_l2_loss(u, tau):
    sign = (u < 0).float()
    weight = (1 - sign) * tau + sign * (1 - tau)
    return (weight * (u**2)).mean()


class IQL(parl.Algorithm):
    def __init__(self,
                 model,
                 max_steps,
                 lr=0.0003,
                 tau=0.7,
                 beta=3.,
                 discount=0.99,
                 alpha=0.005):

        self.device = torch.device("cuda" if torch.cuda.
                                   is_available() else "cpu")
        self.model = model.to(self.device)
        self.q_target = copy.deepcopy(self.model).to(self.device)
        self.lr = lr

        self.v_optimizer = torch.optim.Adam(
            self.model.get_value_params(), lr=self.lr)
        self.q_optimizer = torch.optim.Adam(
            self.model.get_critic_params(), lr=self.lr)
        self.policy_optimizer = torch.optim.Adam(
            self.model.get_actor_params(), lr=self.lr)
        
        self.tau = tau
        self.beta = beta
        self.discount = discount
        self.alpha = alpha

    def predict(self, observations):
        act = self.model.actor_model.act(observations, deterministic=True)
        return act

    def update(self, observations, actions, rewards, next_observations,
               terminals):
        with torch.no_grad():
            target_q1, target_q2 = self.q_target.qvalue(observations, actions)
            target_q = torch.min(target_q1, target_q2)
            next_v = self.model.value(next_observations)

        # Update value function
        v = self.model.value(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.tau)

        # Update Q function
        targets = rewards + (
            1. - terminals.float()) * self.discount * next_v.detach()
        q1, q2 = self.model.qvalue(observations, actions)
        qf1_loss = F.mse_loss(q1, targets)
        qf2_loss = F.mse_loss(q2, targets)
        q_loss = qf1_loss + qf2_loss

        # Update target Q network

        # Update policy
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.model.policy(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions)
        policy_loss = torch.mean(exp_adv * bc_losses)

        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()

        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()

        self.sync_target(alpha=self.alpha)
        return q_loss.cpu().detach(), v_loss.cpu().detach(), policy_loss.cpu(
        ).detach()

    def sync_target(self, alpha=0):
        """ update the target network with the training network
        Args:
            decay(float): the decaying factor while updating the target network with the training network.
                        0 represents the **assignment**. None represents updating the target network slowly that depends on the hyperparameter `tau`.
        """

        for param, target_param in zip(self.model.parameters(),
                                       self.q_target.parameters()):
            target_param.data.copy_(alpha * param.data +
                                    (1 - alpha) * target_param.data)
        return None
