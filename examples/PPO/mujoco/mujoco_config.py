#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

mujoco_config = {
    ## Commented parameters are set to default values in ppo

    #==========  env config ==========
    'env': 'HalfCheetah-v2',  # environment name
    'env_num': 5,  # number of the environment
    'seed': 120,  # seed of the experiment
    'xparl_addr': "localhost:8010",  # xparl address for distributed training

    #==========  training config ==========
    'train_total_episodes': int(1e6),  # max training steps
    'episodes_per_batch': 5,
    'policy_learn_times': 20,  # number of epochs for updating (ie K in the paper)
    'value_learn_times': 10,
    'value_batch_size': 256,
    'eval_episode': 3,
    'test_every_episodes': int(5e3),  # interval between evaluations

    #========== coefficient of ppo ==========
    'kl_targ': 0.003,  # D_KL target value
    'loss_type': 'KLPEN',  # Choose loss type of PPO algorithm, 'CLIP' or 'KLPEN'
    'eps': 1e-5,  # Adam optimizer epsilon (default: 1e-5)
    'clip_param': 0.2,  # epsilon in clipping loss
    'gamma': 0.995,  # discounting factor
    'gae_lambda': 0.98,  # Lambda parameter for calculating N-step advantage
}
