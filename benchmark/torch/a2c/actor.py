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

import os
import gym
import parl
import torch
import numpy as np
from collections import defaultdict
from parl.env.atari_wrappers import wrap_deepmind, MonitorEnv, get_wrapper_by_cls
from parl.env.vector_env import VectorEnv
from parl.utils.rl_utils import calc_gae

from atari_model import ActorCritic
from parl.algorithms import A2C
from atari_agent import Agent


@parl.remote_class(wait=False)
class Actor(object):
    def __init__(self, config):
        # the cluster may not have gpu
        self.device = torch.device("cuda" if torch.cuda.
                                   is_available() else "cpu")
        self.config = config
        self.envs = []
        for _ in range(config['env_num']):
            env = gym.make(config['env_name'])
            env = wrap_deepmind(env, dim=config['env_dim'], obs_format='NCHW')
            self.envs.append(env)
        self.vector_env = VectorEnv(self.envs)

        self.obs_batch = self.vector_env.reset()

        obs_shape = env.observation_space.shape
        act_dim = env.action_space.n

        self.config['obs_shape'] = obs_shape
        self.config['act_dim'] = act_dim

        model = ActorCritic(act_dim)

        model = model.to(self.device)

        algorithm = A2C(model, config)
        self.agent = Agent(algorithm, config)

    def sample(self):
        ''' Interact with the environments lambda times
        '''
        sample_data = defaultdict(list)

        env_sample_data = {}
        for env_id in range(self.config['env_num']):
            env_sample_data[env_id] = defaultdict(list)
        for i in range(self.config['sample_batch_steps']):
            self.obs_batch = np.stack(self.obs_batch)
            action_batch, value_batch = self.agent.sample(self.obs_batch)
            next_obs_batch, reward_batch, done_batch, info_batch = self.vector_env.step(
                action_batch)

            for env_id in range(self.config['env_num']):
                env_sample_data[env_id]['obs'].append(self.obs_batch[env_id])
                env_sample_data[env_id]['actions'].append(
                    action_batch[env_id].item())
                env_sample_data[env_id]['rewards'].append(reward_batch[env_id])
                env_sample_data[env_id]['dones'].append(done_batch[env_id])
                env_sample_data[env_id]['values'].append(
                    value_batch[env_id].item())

                if done_batch[
                        env_id] or i == self.config['sample_batch_steps'] - 1:
                    next_value = 0
                    if not done_batch[env_id]:
                        next_obs = np.expand_dims(next_obs_batch[env_id], 0)
                        next_value = self.agent.value(next_obs).item()

                    values = env_sample_data[env_id]['values']
                    rewards = env_sample_data[env_id]['rewards']
                    advantages = calc_gae(rewards, values, next_value,
                                          self.config['gamma'],
                                          self.config['lambda'])
                    target_values = advantages + values

                    sample_data['obs'].extend(env_sample_data[env_id]['obs'])
                    sample_data['actions'].extend(
                        env_sample_data[env_id]['actions'])
                    sample_data['advantages'].extend(advantages)
                    sample_data['target_values'].extend(target_values)

                    env_sample_data[env_id] = defaultdict(list)

            self.obs_batch = next_obs_batch

        for key in sample_data:
            sample_data[key] = np.stack(sample_data[key])

        return sample_data

    def get_metrics(self):
        metrics = defaultdict(list)
        for env in self.envs:
            monitor = get_wrapper_by_cls(env, MonitorEnv)
            if monitor is not None:
                for episode_rewards, episode_steps in monitor.next_episode_results(
                ):
                    metrics['episode_rewards'].append(episode_rewards)
                    metrics['episode_steps'].append(episode_steps)
        return metrics

    def set_weights(self, params):
        self.agent.set_weights(params)
