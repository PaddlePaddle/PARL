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

import parl
import carla
import gym
import gym_carla
from parl.env.continuous_wrappers import ActionMappingWrapper


@parl.remote_class(wait=False)
class CarlaEnv(object):
    def __init__(self, port):
        class ActionSpace(object):
            def __init__(self,
                         action_space=None,
                         low=None,
                         high=None,
                         shape=None,
                         n=None):
                self.action_space = action_space
                self.low = low
                self.high = high
                self.shape = shape
                self.n = n

            def sample(self):
                return self.action_space.sample()

        params = {
            'obs_size': (160, 100),  # screen size of cv2 window
            'dt': 0.025,  # time interval between two frames
            'ego_vehicle_filter':
            'vehicle.lincoln*',  # filter for defining ego vehicle
            'port': port,  # connection port
            'task_mode':
            'Lane',  # mode of the task, [random, roundabout (only for Town03)]
            'code_mode': 'train',
            'max_time_episode': 250,  # maximum timesteps per episode
            'desired_speed': 15,  # desired speed (m/s)
            'max_ego_spawn_times': 100,  # maximum times to spawn ego vehicle
        }
        self.env = gym.make('carla-v0', params=params)
        self.env = ActionMappingWrapper(self.env)
        self.env.seed(port)
        self.action_space = ActionSpace(
            self.env.action_space, self.env.action_space.low,
            self.env.action_space.high, self.env.action_space.shape)

    def reset(self):
        obs, _ = self.env.reset()
        return obs

    def step(self, action):
        return self.env.step(action)
