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

from copy import deepcopy
# set max_episode_steps according to task_mode
# e.g. task_model   max_episode_steps
#      Lane         250
#      Long         200
TASK_MODE = 'Lane'
MAX_EPISODE_STEPS = 250
params = {
    'obs_size': (160, 100),  # screen size of cv2 window
    'dt': 0.025,  # time interval between two frames
    'ego_vehicle_filter':
    'vehicle.lincoln*',  # filter for defining ego vehicle
    'port': 2000,  # CARLA service's port
    'task_mode':
    TASK_MODE,  # mode of the task, [random, roundabout (only for Town03)]
    'code_mode': 'test',
    'max_time_episode': MAX_EPISODE_STEPS,  # maximum timesteps per episode
    'desired_speed': 15,  # desired speed (m/s)
    'max_ego_spawn_times': 100,  # maximum times to spawn ego vehicle
}

# train env params
train_env_ports = [2021, 2023, 2025]
train_code_mode = 'train'
train_envs_params = []
for port in train_env_ports:
    temp_params = deepcopy(params)
    temp_params['port'] = port
    temp_params['code_mode'] = train_code_mode
    train_envs_params.append(temp_params)

# evaluate env params
eval_port = 2027
eval_code_mode = 'test'
temp_params = deepcopy(params)
temp_params['port'] = eval_port
temp_params['code_mode'] = eval_code_mode
eval_env_params = temp_params

EnvConfig = {
    # train envs config
    'train_envs_params': train_envs_params,

    # eval env config
    'eval_env_params': eval_env_params,
}
