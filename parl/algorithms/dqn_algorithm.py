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

import paddle.fluid as fluid
from parl.framework.algorithm_base import Algorithm
import parl.layers as layers
import copy


class DQNAlgorithm(Algorithm):
    def __init__(self, model, hyperparas):
        Algorithm.__init__(self, model, hyperparas)
        self.model = model
        self.target_model = copy.deepcopy(model)
        # fetch hyper parameters
        self.action_dim = hyperparas['action_dim']
        self.gamma = hyperparas['gamma']
        self.lr = hyperparas['lr']

    def define_predict(self, obs):
        return self.model.value(obs)

    def define_learn(self, obs, action, reward, next_obs, terminal):
        pred_value = self.model.value(obs)
        #fluid.layers.Print(pred_value, summarize=10, message='pred_value')
        next_pred_value = self.target_model.value(next_obs)
        #fluid.layers.Print(next_pred_value, summarize=10, message='next_pred_value')
        best_v = layers.reduce_max(next_pred_value, dim=1)
        best_v.stop_gradient = True
        #fluid.layers.Print(best_v, summarize=10, message='best_v')
        target = reward + (
            1.0 - layers.cast(terminal, dtype='float32')) * self.gamma * best_v

        #fluid.layers.Print(target, summarize=10, message='target')
        action_onehot = layers.one_hot(action, self.action_dim)
        action_onehot = layers.cast(action_onehot, dtype='float32')
        pred_action_value = layers.reduce_sum(
            layers.elementwise_mul(action_onehot, pred_value), dim=1)
        #fluid.layers.Print(pred_action_value, summarize=10, message='pred_action_value')
        cost = layers.square_error_cost(pred_action_value, target)
        cost = layers.reduce_mean(cost)
        optimizer = fluid.optimizer.Adam(self.lr * 0.5, epsilon=1e-3)
        optimizer.minimize(cost)
        return cost

    def sync_target(self, gpu_id):
        self.model.sync_params_to(self.target_model, gpu_id=gpu_id)
