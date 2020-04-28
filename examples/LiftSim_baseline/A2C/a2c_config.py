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

config = {
    #==========  remote config ==========
    'master_address': 'localhost:8010',

    #==========  actor config ==========
    'actor_num': 5,
    'env_num': 5,
    'sample_batch_steps': 5,

    #==========  learner config ==========
    'max_sample_steps': int(1e10),
    'gamma': 0.998,
    'lambda': 1.0,  # GAE

    # start learning rate
    'start_lr': 1.0e-4,

    # coefficient of policy entropy adjustment schedule: (train_step, coefficient)
    'entropy_coeff_scheduler': [(0, -2.0e-4)],
    'vf_loss_coeff': 0.5,
    'get_remote_metrics_interval': 100,
    'log_metrics_interval_s': 60,
}
