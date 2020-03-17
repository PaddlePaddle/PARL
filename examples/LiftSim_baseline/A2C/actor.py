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

import numpy as np
import parl
from collections import defaultdict
from env_wrapper import ObsProcessWrapper, ActionProcessWrapper, RewardWrapper, MetricsWrapper
from parl.utils.rl_utils import calc_gae
from parl.env.vector_env import VectorEnv
from rlschool import LiftSim
from copy import deepcopy
from lift_model import LiftModel
from lift_agent import LiftAgent


@parl.remote_class
class Actor(object):
    def __init__(self, config):
        self.config = config
        self.env_num = config['env_num']

        self.envs = []
        for _ in range(self.env_num):
            env = LiftSim()
            env = RewardWrapper(env)
            env = ActionProcessWrapper(env)
            env = ObsProcessWrapper(env)
            env = MetricsWrapper(env)
            self.envs.append(env)
        self.vector_env = VectorEnv(self.envs)

        # number of elevators
        self.ele_num = self.envs[0].mansion_attr.ElevatorNumber

        act_dim = self.envs[0].act_dim
        self.obs_dim = self.envs[0].obs_dim
        self.config['obs_dim'] = self.obs_dim

        # nested list of shape (env_num, ele_num, obs_dim)
        self.obs_batch = self.vector_env.reset()
        # (env_num * ele_num, obs_dim)
        self.obs_batch = np.array(self.obs_batch).reshape(
            [self.env_num * self.ele_num, self.obs_dim])

        model = LiftModel(act_dim)
        algorithm = parl.algorithms.A3C(
            model, vf_loss_coeff=config['vf_loss_coeff'])
        self.agent = LiftAgent(algorithm, config)

    def sample(self):
        sample_data = defaultdict(list)

        env_sample_data = {}
        # treat each elevator in Liftsim as an independent env
        for env_id in range(self.env_num * self.ele_num):
            env_sample_data[env_id] = defaultdict(list)

        for i in range(self.config['sample_batch_steps']):
            actions_batch, values_batch = self.agent.sample(self.obs_batch)

            vector_actions = np.array_split(actions_batch, self.env_num)
            assert len(vector_actions[-1]) == self.ele_num
            next_obs_batch, reward_batch, done_batch, info_batch = \
                    self.vector_env.step(vector_actions)

            # (env_num, ele_num, obs_dim) -> (env_num * ele_num, obs_dim)
            next_obs_batch = np.array(next_obs_batch).reshape(
                [self.env_num * self.ele_num, self.obs_dim])
            # repeat reward and done to ele_num times
            # (env_num) -> (env_num, ele_num) -> (env_num * ele_num)
            reward_batch = np.repeat(reward_batch, self.ele_num)
            done_batch = np.repeat(done_batch, self.ele_num)

            for env_id in range(self.env_num * self.ele_num):
                env_sample_data[env_id]['obs'].append(self.obs_batch[env_id])
                env_sample_data[env_id]['actions'].append(
                    actions_batch[env_id])
                env_sample_data[env_id]['rewards'].append(reward_batch[env_id])
                env_sample_data[env_id]['dones'].append(done_batch[env_id])
                env_sample_data[env_id]['values'].append(values_batch[env_id])

                # Calculate advantages when the episode is done or reaches max sample steps.
                if done_batch[env_id] or i + 1 == self.config[
                        'sample_batch_steps']:  # reach max sample steps
                    next_value = 0
                    if not done_batch[env_id]:
                        next_obs = np.expand_dims(next_obs_batch[env_id], 0)
                        next_value = self.agent.value(next_obs)

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

            self.obs_batch = deepcopy(next_obs_batch)

        # size of sample_data[key]: env_num * ele_num * sample_batch_steps
        for key in sample_data:
            sample_data[key] = np.stack(sample_data[key])

        return sample_data

    def get_metrics(self):
        metrics = defaultdict(list)
        for metrics_env in self.envs:
            assert isinstance(
                metrics_env,
                MetricsWrapper), "Put the MetricsWrapper in the last wrapper"
            for env_reward_1h, env_reward_24h in metrics_env.next_episode_results(
            ):
                metrics['env_reward_1h'].append(env_reward_1h)
                metrics['env_reward_24h'].append(env_reward_24h)
        return metrics

    def set_weights(self, params):
        self.agent.set_weights(params)
