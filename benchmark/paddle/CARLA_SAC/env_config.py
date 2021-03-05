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
    # screen size of cv2 window
    'obs_size': (160, 100),
    # time interval between two frames
    'dt': 0.025,
    # filter for defining ego vehicle
    'ego_vehicle_filter': 'vehicle.lincoln*',
    # CARLA service's port
    'port': 2000,
    # mode of the task, [random, roundabout (only for Town03)]
    'task_mode': TASK_MODE,
    # mode of env (test/train)
    'code_mode': 'test',
    # maximum timesteps per episode
    'max_time_episode': MAX_EPISODE_STEPS,
    # desired speed (m/s)
    'desired_speed': 15,
    # maximum times to spawn ego vehicle
    'max_ego_spawn_times': 100,
}

# train env params
"""
Set ports of CARLA services for parallel data collecting and training.
You can start CARLA services in different new terminals with respect to the ports list.

e.g.1
    set three ports --> parallel training with three envs
    train_env_ports = [2021, 2023, 2025]
e.g.2
    set five ports --> parallel training with five envs
    train_env_ports = [2017, 2019, 2021, 2023, 2025]
"""
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

# test env params
test_port = 2029
test_code_mode = 'test'
temp_params = deepcopy(params)
temp_params['port'] = test_port
temp_params['code_mode'] = test_code_mode
test_env_params = temp_params

EnvConfig = {
    'env_name': 'carla-v0',

    # train envs config
    'train_envs_params': train_envs_params,
    'env_num': len(train_envs_params),

    # eval env config
    'eval_env_params': eval_env_params,

    # env config for evaluate.py
    'test_env_params': test_env_params,
}
