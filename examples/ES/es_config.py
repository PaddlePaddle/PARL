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
    'master_address': 'localhost:8037',

    #==========  env config ==========
    'env_name': 'Humanoid-v1',

    #==========  actor config ==========
    'actor_num': 96,
    'action_noise_std': 0.01,
    'min_task_runtime': 0.2,
    'eval_prob': 0.003,
    'filter_update_prob': 0.01,

    #==========  learner config ==========
    'stepsize': 0.01,
    'min_episodes_per_batch': 1000,
    'min_steps_per_batch': 10000,
    'noise_size': 200000000,
    'noise_stdev': 0.02,
    'l2_coeff': 0.005,
    'report_window_size': 10,
}
