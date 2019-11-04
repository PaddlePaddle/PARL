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
from parl import layers
from mlp_model import ActorModel, CriticModel
from paddle import fluid
from parl.utils import logger

VEL_OBS_DIM = 4 + 15
OBS_DIM = 98 + VEL_OBS_DIM
ACT_DIM = 22


class EnsembleBaseModel(object):
    def __init__(self,
                 model_dirname=None,
                 stage_name=None,
                 ensemble_num=12,
                 use_cuda=False):
        self.stage_name = stage_name
        self.ensemble_num = ensemble_num
        self.actors = []
        self.critics1 = []
        self.critics2 = []
        for i in range(ensemble_num):
            self.actors.append(
                ActorModel(
                    OBS_DIM,
                    VEL_OBS_DIM,
                    ACT_DIM,
                    stage_name=stage_name,
                    model_id=i))
            self.critics1.append(
                CriticModel(
                    OBS_DIM,
                    VEL_OBS_DIM,
                    ACT_DIM,
                    stage_name=stage_name,
                    model_id=i * 2))
            self.critics2.append(
                CriticModel(
                    OBS_DIM,
                    VEL_OBS_DIM,
                    ACT_DIM,
                    stage_name=stage_name,
                    model_id=i * 2 + 1))

        self._define_program()

        self.place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        self.fluid_executor = fluid.Executor(self.place)
        self.fluid_executor.run(self.startup_program)

        if model_dirname is not None:
            self._load_params(model_dirname)

    def _load_params(self, dirname):
        logger.info('[{}]: Loading model from {}'.format(
            self.stage_name, dirname))
        fluid.io.load_params(
            executor=self.fluid_executor,
            dirname=dirname,
            main_program=self.ensemble_predict_program,
            filename='model.ckpt')

    def _define_program(self):
        self.ensemble_predict_program = fluid.Program()
        self.startup_program = fluid.Program()
        with fluid.program_guard(self.ensemble_predict_program,
                                 self.startup_program):
            obs = layers.data(name='obs', shape=[OBS_DIM], dtype='float32')
            action = self._ensemble_predict(obs)
            self.ensemble_predict_output = [action]

    def _ensemble_predict(self, obs):
        actor_outputs = []
        for i in range(self.ensemble_num):
            actor_outputs.append(self.actors[i].predict(obs))
        batch_actions = layers.concat(actor_outputs, axis=0)
        batch_obs = layers.expand(obs, expand_times=[self.ensemble_num, 1])

        critic_outputs = []
        for i in range(self.ensemble_num):
            critic1_output = self.critics1[i].predict(batch_obs, batch_actions)
            critic1_output = layers.unsqueeze(critic1_output, axes=[1])

            critic2_output = self.critics2[i].predict(batch_obs, batch_actions)
            critic2_output = layers.unsqueeze(critic2_output, axes=[1])

            critic_output = layers.elementwise_min(critic1_output,
                                                   critic2_output)
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
        ensemble_predict_action = layers.squeeze(
            ensemble_predict_action, axes=[0])
        return ensemble_predict_action

    def pred_batch(self, obs):
        feed = {'obs': obs}
        action = self.fluid_executor.run(
            self.ensemble_predict_program,
            feed=feed,
            fetch_list=self.ensemble_predict_output)[0]
        return action


class SubmitModel(object):
    def __init__(self, use_cuda=False):
        self.stage0_model = EnsembleBaseModel(
            model_dirname='./stage0_saved_models',
            stage_name='stage0',
            use_cuda=use_cuda)
        self.stage1_model = EnsembleBaseModel(
            model_dirname='./stage1_saved_models',
            stage_name='stage1',
            use_cuda=use_cuda)

    def pred_batch(self, obs, target_change_times):
        batch_obs = np.expand_dims(obs, axis=0).astype('float32')
        if target_change_times == 0:
            action = self.stage0_model.pred_batch(batch_obs)
        else:
            action = self.stage1_model.pred_batch(batch_obs)
        return action


if __name__ == '__main__':
    submit_model = SubmitModel()
