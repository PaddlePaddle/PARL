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
import parl.layers as layers
from mlp_model import ActorModel, CriticModel
from paddle import fluid
from parl.utils import logger

VEL_OBS_DIM = 4
OBS_DIM = 185 + VEL_OBS_DIM
ACT_DIM = 19


class EnsembleBaseModel(object):
    def __init__(self,
                 model_dirname=None,
                 stage_name=None,
                 ensemble_num=12,
                 use_cuda=False):
        self.stage_name = stage_name
        self.ensemble_num = ensemble_num
        self.actors = []
        self.critics = []
        for i in range(ensemble_num):
            self.actors.append(
                ActorModel(
                    OBS_DIM,
                    VEL_OBS_DIM,
                    ACT_DIM,
                    stage_name=stage_name,
                    model_id=i))
            self.critics.append(
                CriticModel(
                    OBS_DIM,
                    VEL_OBS_DIM,
                    ACT_DIM,
                    stage_name=stage_name,
                    model_id=i))

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
            main_program=self.ensemble_predict_program)

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
            critic_output = self.critics[i].predict(batch_obs, batch_actions)
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


class StartModel(EnsembleBaseModel):
    def __init__(self, use_cuda):
        super(StartModel, self).__init__(
            model_dirname='saved_model',
            stage_name='stage0',
            use_cuda=use_cuda)


class Stage123Model(EnsembleBaseModel):
    def __init__(self, use_cuda):
        super(Stage123Model, self).__init__(
            model_dirname='saved_model',
            stage_name='stage123',
            use_cuda=use_cuda)


class SubmitModel(object):
    def __init__(self, use_cuda=False):
        self.start_model = StartModel(use_cuda=use_cuda)
        self.stage123_model = Stage123Model(use_cuda=use_cuda)

    def pred_batch(self, obs, stage_idx):
        batch_obs = np.expand_dims(obs, axis=0).astype('float32')
        if stage_idx == 0:
            action = self.start_model.pred_batch(batch_obs)
        else:
            action = self.stage123_model.pred_batch(batch_obs)
        return action


if __name__ == '__main__':
    submit_model = SubmitModel()
