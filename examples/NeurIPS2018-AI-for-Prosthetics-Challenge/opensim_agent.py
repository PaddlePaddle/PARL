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
import re
import parl
from parl import layers
from paddle import fluid
from paddle.fluid.executor import _fetch_var
from parl.utils import logger


class OpenSimAgent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim, ensemble_num):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.ensemble_num = ensemble_num
        super(OpenSimAgent, self).__init__(algorithm)

        # Use ParallelExecutor to make program running faster
        use_cuda = True if parl.GPU_ID >= 0 else False
        self.learn_pe = []
        self.pred_pe = []

        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.use_experimental_executor = True
        exec_strategy.num_threads = 4
        build_strategy = fluid.BuildStrategy()
        build_strategy.remove_unnecessary_lock = True

        for i in range(self.ensemble_num):
            with fluid.scope_guard(fluid.global_scope().new_scope()):
                pe = fluid.ParallelExecutor(
                    use_cuda=use_cuda,
                    main_program=self.learn_programs[i],
                    exec_strategy=exec_strategy,
                    build_strategy=build_strategy)
                self.learn_pe.append(pe)

            with fluid.scope_guard(fluid.global_scope().new_scope()):
                pe = fluid.ParallelExecutor(
                    use_cuda=use_cuda,
                    main_program=self.predict_programs[i],
                    exec_strategy=exec_strategy,
                    build_strategy=build_strategy)
                self.pred_pe.append(pe)

            # Attention: In the beginning, sync target model totally.
            self.alg.sync_target(
                model_id=i,
                decay=0,
                share_vars_parallel_executor=self.learn_pe[i])
            # Do cache, will create ParallelExecutor of sync params in advance
            # If not, there are some issues when ensemble_num > 1
            self.alg.sync_target(
                model_id=i, share_vars_parallel_executor=self.learn_pe[i])

        with fluid.scope_guard(fluid.global_scope().new_scope()):
            self.ensemble_predict_pe = fluid.ParallelExecutor(
                use_cuda=use_cuda,
                main_program=self.ensemble_predict_program,
                exec_strategy=exec_strategy,
                build_strategy=build_strategy)

    def build_program(self):
        self.predict_programs = []
        self.predict_outputs = []
        self.learn_programs = []
        self.learn_programs_output = []
        for i in range(self.ensemble_num):
            predict_program = fluid.Program()
            with fluid.program_guard(predict_program):
                obs = layers.data(
                    name='obs', shape=[self.obs_dim], dtype='float32')
                act = self.alg.predict(obs, model_id=i)
            self.predict_programs.append(predict_program)
            self.predict_outputs.append([act.name])

            learn_program = fluid.Program()
            with fluid.program_guard(learn_program):
                obs = layers.data(
                    name='obs', shape=[self.obs_dim], dtype='float32')
                act = layers.data(
                    name='act', shape=[self.act_dim], dtype='float32')
                reward = layers.data(name='reward', shape=[], dtype='float32')
                next_obs = layers.data(
                    name='next_obs', shape=[self.obs_dim], dtype='float32')
                terminal = layers.data(name='terminal', shape=[], dtype='bool')
                actor_lr = layers.data(
                    name='actor_lr',
                    shape=[1],
                    dtype='float32',
                    append_batch_size=False)
                critic_lr = layers.data(
                    name='critic_lr',
                    shape=[1],
                    dtype='float32',
                    append_batch_size=False)
                actor_loss, critic_loss = self.alg.learn(
                    obs,
                    act,
                    reward,
                    next_obs,
                    terminal,
                    actor_lr,
                    critic_lr,
                    model_id=i)
            self.learn_programs.append(learn_program)
            self.learn_programs_output.append([critic_loss.name])

        self.ensemble_predict_program = fluid.Program()
        with fluid.program_guard(self.ensemble_predict_program):
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            act = self.alg.ensemble_predict(obs)
        self.ensemble_predict_output = [act.name]

    def predict(self, obs, model_id):
        feed = {'obs': obs}
        feed = [feed]
        act = self.pred_pe[model_id].run(
            feed=feed, fetch_list=self.predict_outputs[model_id])[0]
        return act

    def ensemble_predict(self, obs):
        feed = {'obs': obs}
        feed = [feed]
        act = self.ensemble_predict_pe.run(
            feed=feed, fetch_list=self.ensemble_predict_output)[0]
        return act

    def learn(self, obs, act, reward, next_obs, terminal, actor_lr, critic_lr,
              model_id):
        feed = {
            'obs': obs,
            'act': act,
            'reward': reward,
            'next_obs': next_obs,
            'terminal': terminal,
            'actor_lr': actor_lr,
            'critic_lr': critic_lr
        }

        feed = [feed]
        critic_loss = self.learn_pe[model_id].run(
            feed=feed, fetch_list=self.learn_programs_output[model_id])[0]
        self.alg.sync_target(
            model_id=model_id,
            share_vars_parallel_executor=self.learn_pe[model_id])
        return critic_loss

    def save_params(self, dirname):
        for i in range(self.ensemble_num):
            fluid.io.save_params(
                executor=self.fluid_executor,
                dirname=dirname,
                main_program=self.learn_programs[i])

    def load_params(self, dirname, from_one_head):
        if from_one_head:
            logger.info('[From one head, extend to multi head:]')
            # load model 0
            fluid.io.load_params(
                executor=self.fluid_executor,
                dirname=dirname,
                main_program=self.learn_programs[0])

            # sync identity params of model/target_model 0 to other models/target_models
            for i in range(1, self.ensemble_num):
                params = list(
                    filter(
                        lambda x: 'identity' in x.name and '@GRAD' not in x.name,
                        self.learn_programs[i].list_vars()))
                for param in params:
                    param_var = _fetch_var(param.name, return_numpy=False)

                    model0_name = re.sub(r"identity_\d+", "identity_0",
                                         param.name)
                    model0_value = _fetch_var(model0_name, return_numpy=True)
                    logger.info('{} -> {}'.format(model0_name, param.name))
                    param_var.set(model0_value, self.place)

            # sync share params of target_model 0 to other target models
            # After deepcopy, shapre params between target models  is different
            for i in range(1, self.ensemble_num):
                params = list(
                    filter(
                        lambda x: 'shared' in x.name and 'PARL_target' in x.name and '@GRAD' not in x.name,
                        self.learn_programs[i].list_vars()))
                for param in params:
                    param_var = _fetch_var(param.name, return_numpy=False)

                    model0_name = re.sub(r"_\d+$", "_0", param.name)
                    model0_value = _fetch_var(model0_name, return_numpy=True)
                    logger.info('{} -> {}'.format(model0_name, param.name))
                    param_var.set(model0_value, self.place)

        else:
            for i in range(self.ensemble_num):
                fluid.io.load_params(
                    executor=self.fluid_executor,
                    dirname=dirname,
                    main_program=self.learn_programs[i])
