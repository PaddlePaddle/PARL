#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

    # configuration for env
    "board_size": 21,

    # configuration for training
    "episodes": 10000,
    "batch_size": 128,
    "train_times": 2,
    "gamma": 0.997,
    "lr": 0.0001,
    "test_every_episode": 100,

    # configuration for ppo algorithm
    "vf_loss_coef": 1,
    "ent_coef": 0.01,

    # configuration for the observation of ships
    "world_dim": 5 * 21 * 21,
    "ship_obs_dim": 6,
    "ship_act_dim": 5,
    "ship_max_step": 10000,

    # the number of halite we want the ships to obtain (e.g K)
    "num_halite": 100,

    # the maximum number of ships (e.g M)
    "num_ships": 10,

    # seed for training
    "seed": 123456,

    # configuration for logging
    "log_path": './train_log/',
    "save_path": './save_model/',
}
