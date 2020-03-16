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

import numpy as np
import paddle.fluid as fluid
import parl
import utils
from parl import layers
from optimizers import Adam


class MujocoAgent(parl.Agent):
    def __init__(self, algorithm, config):
        self.config = config
        super(MujocoAgent, self).__init__(algorithm)

        weights = self.get_weights()
        assert len(
            weights) == 1, "There should be only one model in the algorithm."
        self.weights_name = list(weights.keys())[0]
        weights = list(weights.values())[0]
        self.weights_shapes = [x.shape for x in weights]
        self.weights_total_size = np.sum(
            [np.prod(x) for x in self.weights_shapes])

        self.optimizer = Adam(self.weights_total_size, self.config['stepsize'])

    def build_program(self):
        self.predict_program = fluid.Program()

        with fluid.program_guard(self.predict_program):
            obs = layers.data(
                name='obs', shape=[self.config['obs_dim']], dtype='float32')
            self.predict_action = self.alg.predict(obs)
        self.predict_program = parl.compile(self.predict_program)

    def learn(self, noisy_rewards, noises):
        """ Update weights of the model in the numpy level.

        Compute the grident and take a step.

        Args:
            noisy_rewards(np.float32): [batch_size, 2]
            noises(np.float32): [batch_size, weights_total_size]
        """

        g = utils.batched_weighted_sum(
            # mirrored sampling: evaluate pairs of perturbations \epsilon, −\epsilon
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

    def predict(self, obs):
        obs = obs.astype('float32')
        obs = np.expand_dims(obs, axis=0)
        act = self.fluid_executor.run(
            program=self.predict_program,
            feed={'obs': obs},
            fetch_list=[self.predict_action])[0]
        return act

    def get_flat_weights(self):
        weights = list(self.get_weights().values())[0]
        flat_weights = np.concatenate([x.flatten() for x in weights])
        return flat_weights

    def set_flat_weights(self, flat_weights):
        weights = utils.unflatten(flat_weights, self.weights_shapes)
        self.set_weights({self.weights_name: weights})
