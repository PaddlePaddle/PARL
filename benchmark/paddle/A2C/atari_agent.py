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

import numpy as np
import parl
from parl.utils import machine_info
from parl.utils.scheduler import PiecewiseScheduler, LinearDecayScheduler
import paddle


class AtariAgent(parl.Agent):
    def __init__(self, algorithm, config):
        """

        Args:
            algorithm (`parl.Algorithm`): algorithm to be used in this agent.
            config (dict): config file describing the training hyper-parameters(see a2c_config.py)
        """

        self.obs_shape = config['obs_shape']
        super(AtariAgent, self).__init__(algorithm)

        self.lr_scheduler = LinearDecayScheduler(config['start_lr'],
                                                 config['max_sample_steps'])

        self.entropy_coeff_scheduler = PiecewiseScheduler(
            config['entropy_coeff_scheduler'])

    def sample(self, obs_np):
        """
        Args:
            obs_np: a numpy float32 array of shape ([B] + observation_space).
                    Format of image input should be NCHW format.

        Returns:
            sample_actions: a numpy  int64 array of shape [B]
            values: a numpy float32 array of shape [B]
        """
        obs_np = paddle.to_tensor(obs_np, dtype='float32')
        probs, values = self.alg.prob_and_value(obs_np)
        probs = probs.cpu().numpy()
        values = values.cpu().numpy()
        sample_actions = np.array(
            [np.random.choice(len(prob), 1, p=prob)[0] for prob in probs])
        return sample_actions, values

    def predict(self, obs_np):
        """
        Args:
            obs_np: a numpy float32 array of shape ([B] + observation_space).
                    Format of image input should be NCHW format.

        Returns:
            predict_actions: a numpy int64 array of shape [B]
        """
        obs_np = paddle.to_tensor(obs_np, dtype='float32')
        predict_actions = self.alg.predict(obs_np)
        return predict_actions.cpu().numpy()

    def value(self, obs_np):
        """
        Args:
            obs_np: a numpy float32 array of shape ([B] + observation_space).
                    Format of image input should be NCHW format.
        Returns:
            values: a numpy float32 array of shape [B]
        """
        obs_np = paddle.to_tensor(obs_np, dtype='float32')
        values = self.alg.value(obs_np)

        return values.cpu().numpy()

    def learn(self, obs_np, actions_np, advantages_np, target_values_np):
        """
        Args:
            obs_np: a numpy float32 array of shape ([B] + observation_space).
                    Format of image input should be NCHW format.
            actions_np: a numpy int64 array of shape [B]
            advantages_np: a numpy float32 array of shape [B]
            target_values_np: a numpy float32 array of shape [B]
        """

        obs_np = paddle.to_tensor(obs_np, dtype='float32')
        actions_np = paddle.to_tensor(actions_np, dtype='int64')
        advantages_np = paddle.to_tensor(advantages_np, dtype='float32')
        target_values_np = paddle.to_tensor(target_values_np, dtype='float32')

        lr = self.lr_scheduler.step(step_num=obs_np.shape[0])
        entropy_coeff = self.entropy_coeff_scheduler.step()
        total_loss, pi_loss, vf_loss, entropy = self.alg.learn(
            obs_np, actions_np, advantages_np, target_values_np, lr,
            entropy_coeff)

        return total_loss.cpu().numpy(), pi_loss.cpu().numpy(), vf_loss.cpu(
        ).numpy(), entropy.cpu().numpy(), lr, entropy_coeff
