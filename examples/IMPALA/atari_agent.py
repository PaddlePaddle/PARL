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

import numpy as np
import parl
from parl.utils.scheduler import PiecewiseScheduler, LinearDecayScheduler
import paddle


class AtariAgent(parl.Agent):
    def __init__(self, algorithm):
        super(AtariAgent, self).__init__(algorithm)

    def sample(self, obs_np):
        """
        Args:
            obs_np: a numpy float32 array of shape ([B] + observation_space).
            Format of image input should be NCHW format.

        Returns:
            predict_actions: a numpy int64 array of shape [B]
            behaviour_logits: a numpy float32 array of shape [B, act_dim]
        """
        obs = paddle.to_tensor(obs_np, dtype='float32')
        probs, behaviour_logits = self.alg.sample(obs)

        probs = probs.cpu().numpy()
        sample_actions = np.array(
            [np.random.choice(len(prob), 1, p=prob)[0] for prob in probs])

        return sample_actions, behaviour_logits.cpu().numpy()

    def learn(self, obs_np, actions_np, behaviour_logits_np, rewards_np,
              dones_np, lr, entropy_coeff):
        """
        Args:
            obs_np: a numpy float32 array of shape ([B] + observation_space).
                    Format of image input should be NCHW format.
            actions_np: a numpy int64 array of shape [B]
            behaviour_logits_np: a numpy float32 array of shape [B, act_dim]
            rewards_np: a numpy float32 array of shape [B]
            dones_np: a numpy bool array of shape [B]
            lr: float scalar of learning rate.
            entropy_coeff: float scalar of entropy coefficient.
        """

        obs = paddle.to_tensor(obs_np, dtype='float32')
        actions = paddle.to_tensor(actions_np, dtype='int64')
        behaviour_logits = paddle.to_tensor(
            behaviour_logits_np, dtype='float32')
        rewards = paddle.to_tensor(rewards_np, dtype='float32')
        dones = paddle.to_tensor(dones_np, dtype='bool')

        vtrace_loss, kl = self.alg.learn(obs, actions, behaviour_logits,
                                         rewards, dones, lr, entropy_coeff)

        total_loss = vtrace_loss.total_loss.cpu().numpy()
        pi_loss = vtrace_loss.pi_loss.cpu().numpy()
        vf_loss = vtrace_loss.vf_loss.cpu().numpy()
        entropy = vtrace_loss.entropy.cpu().numpy()
        kl = kl.cpu().numpy()

        return total_loss, pi_loss, vf_loss, entropy, kl
