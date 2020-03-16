#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

# arguments of coma
config = {
    # ========== Environment ==========
    'difficulty': '3',  # The difficulty of the game
    'map': '3m',  # The map of the game
    'env_seed': None,  # Environment random seed
    'replay_dir': '',  # Save the replay, not available in Ubuntu

    # ========== Learn ==========
    'gamma': 0.99,
    'grad_norm_clip': 10,  # Prevent gradient explosion
    'td_lambda': 0.8,  # Lambda of td-lambda return
    'actor_lr': 1e-4,
    'critic_lr': 1e-3,
    'target_update_cycle': 200,  # How often to update the target_net

    # ========== Epsilon-greedy ==========
    'epsilon': 0.5,
    'anneal_epsilon': 0.00064,
    'min_epsilon': 0.02,
    # 'epsilon_anneal_scale' : 'epoch',

    # ========== Other ==========
    'n_epoch': 5000,  # The number of the epoch to train the agent
    'n_episodes': 5,  # The number of the episodes in one epoch
    'test_episode_n': 20,  # The Number of the epochs to evaluate the agent
    'threshold': 19,  # The threshold to judge whether win
    'test_cycle': 5,  # How often to evaluate (every 'test_cycle' epcho)
    'save_cycle': 1000,  # How often to save the model
    'model_dir': './model',  # The model directory of the policy
    'test': False,  # Evaluate model and quit (no training)
    'restore': False  # restore model or not
}
