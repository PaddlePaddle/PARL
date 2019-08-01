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

import parl
from parl import layers
from copy import deepcopy
from paddle import fluid

__all__ = ['MultiHeadDDPG']


class MultiHeadDDPG(parl.Algorithm):
    def __init__(self, models, hyperparas):
        """ model: should implement the function get_actor_params()
        """
        self.models = models
        self.target_models = []
        for model in models:
            target_model = deepcopy(model)
            self.target_models.append(target_model)

        # fetch hyper parameters
        self.gamma = hyperparas['gamma']
        self.tau = hyperparas['tau']
        self.ensemble_num = hyperparas['ensemble_num']

    def predict(self, obs, model_id):
        """ use actor model of self.models[model_id] to predict the action
        """
        return self.models[model_id].policy(obs)

    def ensemble_predict(self, obs):
        """ ensemble predict:
        1. For actions of all actors, each critic will score them
           and normalize its scores;
        2. For each actor, will calculate its score by 
           average scores given by all critics
        3. choose action of the actor whose score is best
        """
        actor_outputs = []
        for i in range(self.ensemble_num):
            actor_outputs.append(self.models[i].policy(obs))
        batch_actions = layers.concat(actor_outputs, axis=0)
        batch_obs = layers.expand(obs, expand_times=[self.ensemble_num, 1])

        critic_outputs = []
        for i in range(self.ensemble_num):
            critic_output = self.models[i].value(batch_obs, batch_actions)
            critic_output = layers.unsqueeze(critic_output, axes=[1])
            critic_outputs.append(critic_output)
        score_matrix = layers.concat(critic_outputs, axis=1)

        # Normalize scores given by each critic
        sum_critic_score = layers.reduce_sum(
            score_matrix, dim=0, keep_dim=True)
        sum_critic_score = layers.expand(
            sum_critic_score, expand_times=[self.ensemble_num, 1])
        norm_score_matrix = score_matrix / sum_critic_score

        actions_mean_score = layers.reduce_mean(
            norm_score_matrix, dim=1, keep_dim=True)
        best_score_id = layers.argmax(actions_mean_score, axis=0)
        best_score_id = layers.cast(best_score_id, dtype='int32')
        ensemble_predict_action = layers.gather(batch_actions, best_score_id)
        return ensemble_predict_action

    def learn(self, obs, action, reward, next_obs, terminal, actor_lr,
              critic_lr, model_id):
        """ update actor and critic model of self.models[model_id] with DDPG algorithm
        """
        actor_cost = self._actor_learn(obs, actor_lr, model_id)
        critic_cost = self._critic_learn(obs, action, reward, next_obs,
                                         terminal, critic_lr, model_id)
        return actor_cost, critic_cost

    def _actor_learn(self, obs, actor_lr, model_id):
        action = self.models[model_id].policy(obs)
        Q = self.models[model_id].value(obs, action)
        cost = layers.reduce_mean(-1.0 * Q)
        optimizer = fluid.optimizer.AdamOptimizer(actor_lr)
        optimizer.minimize(
            cost, parameter_list=self.models[model_id].get_actor_params())
        return cost

    def _critic_learn(self, obs, action, reward, next_obs, terminal, critic_lr,
                      model_id):
        next_action = self.target_models[model_id].policy(next_obs)
        next_Q = self.target_models[model_id].value(next_obs, next_action)

        terminal = layers.cast(terminal, dtype='float32')
        target_Q = reward + (1.0 - terminal) * self.gamma * next_Q
        target_Q.stop_gradient = True

        Q = self.models[model_id].value(obs, action)
        cost = layers.square_error_cost(Q, target_Q)
        cost = layers.reduce_mean(cost)
        optimizer = fluid.optimizer.AdamOptimizer(critic_lr)
        optimizer.minimize(cost)
        return cost

    def sync_target(self,
                    model_id,
                    decay=None,
                    share_vars_parallel_executor=None):
        if decay is None:
            decay = 1.0 - self.tau
        self.models[model_id].sync_weights_to(
            self.target_models[model_id],
            decay=decay,
            share_vars_parallel_executor=share_vars_parallel_executor)
