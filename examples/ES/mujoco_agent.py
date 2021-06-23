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
import paddle
import parl
import utils
from optimizers import Adam


class MujocoAgent(parl.Agent):
    def __init__(self, algorithm, config):
        self.config = config
        super(MujocoAgent, self).__init__(algorithm)
        self.obs_shape = config['obs_dim']
        weights = self.get_weights()
        self.weights_name = list(weights.keys())
        weights = list(weights.values())
        self.weights_shapes = [x.shape for x in weights]
        self.weights_total_size = np.sum(
            [np.prod(x) for x in self.weights_shapes])
        self.optimizer = Adam(self.weights_total_size, self.config['stepsize'])

    def predict(self, obs):
        obs = obs.astype('float32')
        obs = np.expand_dims(obs, axis=0)
        obs = paddle.to_tensor(obs, dtype='float32')
        predict_actions = self.alg.predict(obs)
        return predict_actions.detach().numpy()

    def learn(self, noisy_rewards, noises):
        """ Update weights of the model in the numpy level.

        Compute the grident and take a step.

        Args:
            noisy_rewards(np.float32): [batch_size, 2]
            noises(np.float32): [batch_size, weights_total_size]
        """

        g = utils.batched_weighted_sum(
            # Mirrored sampling: evaluate pairs of perturbations \epsilon, −\epsilon
            noisy_rewards[:, 0] - noisy_rewards[:, 1],
            noises,
            batch_size=500)
        g /= noisy_rewards.size

        latest_flat_weights = self.get_flat_weights()
        # Compute the new weights theta.
        theta, update_ratio = self.optimizer.update(
            latest_flat_weights,
            -g + self.config["l2_coeff"] * latest_flat_weights)
        self.set_flat_weights(theta)

    def get_flat_weights(self):
        weights = list(self.get_weights().values())
        flat_weights = np.concatenate([x.flatten() for x in weights])
        return flat_weights

    def set_flat_weights(self, flat_weights):
        weights = utils.unflatten(flat_weights, self.weights_shapes)
        weights_dcit = {}
        assert len(weights) == len(self.weights_name)
        for name, values in zip(self.weights_name, weights):
            weights_dcit[name] = values
        self.set_weights(weights_dcit)
