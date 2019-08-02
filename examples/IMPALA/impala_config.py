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
    'experiment_name': 'Pong',

    #==========  remote config ==========
    'master_address': 'localhost:8010',

    #==========  env config ==========
    'env_name': 'PongNoFrameskip-v4',
    'env_dim': 42,

    #==========  actor config ==========
    'actor_num': 32,
    'env_num': 5,
    'sample_batch_steps': 50,

    #==========  learner config ==========
    'train_batch_size': 1000,
    'sample_queue_max_size': 8,
    'gamma': 0.99,

    # learning rate adjustment schedule: (train_step, learning_rate)
    'lr_scheduler': [(0, 0.001), (20000, 0.0005), (40000, 0.0001)],

    # coefficient of policy entropy adjustment schedule: (train_step, coefficient)
    'entropy_coeff_scheduler': [(0, -0.01)],
    'vf_loss_coeff': 0.5,
    'clip_rho_threshold': 1.0,
    'clip_pg_rho_threshold': 1.0,
    'get_remote_metrics_interval': 10,
    'log_metrics_interval_s': 10,
    'params_broadcast_interval': 5,
}
