#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

atari_config = {
    #==========  env config ==========
    'env_name': 'PongNoFrameskip-v4',
    'continuous_action': False,
    'env_num': 8,

    #==========  training config ==========
    'train_total_steps': int(1e7),  # max training steps
    'step_nums': 128,  # data collecting time steps (ie. T in the paper)
    'num_minibatches': 4,
    'update_epochs':
    4,  # number of epochs for updating using each T data (ie K in the paper)
    'eval_episode': 3,

    #========== coefficient of ppo ==========
    'gamma': 0.99,
    'gae': True,  # whether or not to use GAE
    'gae_lambda': 0.95,  # Lambda parameter for calculating N-step advantage
    'start_lr': 2.5e-4,  # start learning rate
    'lr_decay': True,  # whether or not to use linear decay rl
    'eps': 1e-5,  # Adam optimizer epsilon (default: 1e-5)
    'clip_coef':
    0.1,  # epsilon in clipping loss (ie. clip(r_t, 1 - epsilon, 1 + epsilon))
    'ent_coef': 0.01,  # Entropy coefficient (ie. c_2 in the paper)
    'vf_coef': 0.5,  # Value loss coefficient (ie. c_1 in the paper)
    'max_grad_norm': 0.5,  # Max gradient norm for gradient clipping
    'norm_adv': True,  # advantages normalization
    'clip_vloss':
    True,  # whether or not to use a clipped loss for the value function
}

mujoco_config = {
    #==========  env config ==========
    'env_name': 'HalfCheetah-v2',
    'continuous_action': True,
    'env_num': 1,

    #==========  training config ==========
    'train_total_steps': int(1e6),
    'step_nums': 2048,
    'num_minibatches': 32,
    'update_epochs': 10,
    'start_lr': 3e-4,
    'lr_decay': True,
    'eps': 1e-5,
    'eval_episode': 3,
    #========== coefficient of ppo ==========
    'gamma': 0.99,
    'gae': True,
    'gae_lambda': 0.95,
    'clip_coef': 0.2,
    'ent_coef': 0.0,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'norm_adv': True,
    'clip_vloss': True,
}

Config = {
    'mujoco': mujoco_config,
    'atari': atari_config,
}
