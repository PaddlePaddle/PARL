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

import gym
import numpy as np
import parl
from parl.env.atari_wrappers import wrap_deepmind, MonitorEnv, get_wrapper_by_cls
from collections import defaultdict


@parl.remote_class
class Actor(object):
    def __init__(self, config):
        self.config = config

        env = gym.make(config['env_name'])
        self.env = wrap_deepmind(env, dim=config['env_dim'], obs_format='NCHW')

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done

    def reset(self):
        obs = self.env.reset()
        return obs

    def get_metrics(self):
        metrics = defaultdict(list)
        monitor = get_wrapper_by_cls(self.env, MonitorEnv)
        if monitor is not None:
            for episode_rewards, episode_steps in monitor.next_episode_results(
            ):
                metrics['episode_rewards'].append(episode_rewards)
                metrics['episode_steps'].append(episode_steps)
        return metrics
