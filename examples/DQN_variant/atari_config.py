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
    #==========  env config ==========
    #'env_name': 'PongNoFrameskip-v4',
    'env_name': 'BreakoutNoFrameskip-v4',
    'env_dim': 84,
    #'rom_path': 'rom_files/pong.bin',
    'rom_path': 'rom_files/breakout.bin',
    'train_env_seed': 6,
    'test_env_seed': 16,

    #==========  training config ==========
    'update_target_step': 2500,
    'algorithm': 'DDQN',
    'dueling': True,
    'ep_start': 1,
    'ep_end': 0.1,
    'ep_step': 1000000,
    'batch_size': 32,
    'memory_size': 1000000,
    'gamma': 0.99,
    'lr_start': 0.0003,
    'lr_end': 0.00001,
    'lr_step': 1000000,
    'memory_warmup_size': 50000,
    'update_freq': 4,
    'train_total_steps': 10000000,
    
    #==========  eval and test config ==========
    'eval_episodes': 3,
    'test_episodes': 20,
    'eval_render': False,
    'eval_every_steps': 100000,
}
