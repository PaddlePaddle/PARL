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
import six
import parl
from atari_model import AtariModel
from collections import defaultdict
from atari_agent import AtariAgent
from parl.env.atari_wrappers import wrap_deepmind, MonitorEnv, get_wrapper_by_cls
from parl.env.vector_env import VectorEnv


@parl.remote_class
class Actor(object):
    def __init__(self, config):
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

        model = AtariModel(act_dim)
        algorithm = parl.algorithms.IMPALA(
            model,
            sample_batch_steps=self.config['sample_batch_steps'],
            gamma=self.config['gamma'],
            vf_loss_coeff=self.config['vf_loss_coeff'],
            clip_rho_threshold=self.config['clip_rho_threshold'],
            clip_pg_rho_threshold=self.config['clip_pg_rho_threshold'])
        self.agent = AtariAgent(algorithm, obs_shape, act_dim)

    def sample(self):
        env_sample_data = {}
        for env_id in range(self.config['env_num']):
            env_sample_data[env_id] = defaultdict(list)

        for i in range(self.config['sample_batch_steps']):
            actions, behaviour_logits = self.agent.sample(
                np.stack(self.obs_batch))
            next_obs_batch, reward_batch, done_batch, info_batch = \
                    self.vector_env.step(actions)

            for env_id in range(self.config['env_num']):
                env_sample_data[env_id]['obs'].append(self.obs_batch[env_id])
                env_sample_data[env_id]['actions'].append(actions[env_id])
                env_sample_data[env_id]['behaviour_logits'].append(
                    behaviour_logits[env_id])
                env_sample_data[env_id]['rewards'].append(reward_batch[env_id])
                env_sample_data[env_id]['dones'].append(done_batch[env_id])

            self.obs_batch = next_obs_batch

        # Merge data of envs
        sample_data = defaultdict(list)
        for env_id in range(self.config['env_num']):
            for data_name in [
                    'obs', 'actions', 'behaviour_logits', 'rewards', 'dones'
            ]:
                sample_data[data_name].extend(
                    env_sample_data[env_id][data_name])

        # size of sample_data: env_num * sample_batch_steps
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

    def set_weights(self, weights):
        self.agent.set_weights(weights)
