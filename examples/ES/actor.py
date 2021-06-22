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

import gym
import parl
import time
import numpy as np
from es import ES
from obs_filter import MeanStdFilter
from mujoco_agent import MujocoAgent
from mujoco_model import MujocoModel
from noise import SharedNoiseTable


@parl.remote_class(wait=False)
class Actor(object):
    def __init__(self, config):
        self.config = config

        self.env = gym.make(self.config['env_name'])
        self.config['obs_dim'] = self.env.observation_space.shape[0]
        self.config['act_dim'] = self.env.action_space.shape[0]

        self.obs_filter = MeanStdFilter(self.config['obs_dim'])
        self.noise = SharedNoiseTable(self.config['noise_size'])

        model = MujocoModel(self.config['obs_dim'], self.config['act_dim'])
        algorithm = ES(model)
        self.agent = MujocoAgent(algorithm, self.config)

    def _play_one_episode(self, add_noise=False):
        episode_reward = 0
        episode_step = 0

        obs = self.env.reset()
        while True:
            if np.random.uniform() < self.config['filter_update_prob']:
                obs = self.obs_filter(obs[None], update=True)
            else:
                obs = self.obs_filter(obs[None], update=False)

            action = self.agent.predict(obs)
            if add_noise:
                action += np.random.randn(
                    *action.shape) * self.config['action_noise_std']

            obs, reward, done, _ = self.env.step(action)
            episode_reward += reward
            episode_step += 1
            if done:
                break
        return episode_reward, episode_step

    def sample(self, flat_weights):
        noise_indices, rewards, lengths = [], [], []
        eval_rewards, eval_lengths = [], []

        # Perform some rollouts with noise.
        task_tstart = time.time()
        while (len(noise_indices) == 0
               or time.time() - task_tstart < self.config['min_task_runtime']):

            if np.random.uniform() < self.config["eval_prob"]:
                # Do an evaluation run with no perturbation.
                self.agent.set_flat_weights(flat_weights)
                episode_reward, episode_step = self._play_one_episode(
                    add_noise=False)
                eval_rewards.append(episode_reward)
                eval_lengths.append(episode_step)
            else:
                # Do a regular run with parameter perturbations.
                noise_index = self.noise.sample_index(
                    self.agent.weights_total_size)

                perturbation = self.config["noise_stdev"] * self.noise.get(
                    noise_index, self.agent.weights_total_size)

                # Mirrored sampling: evaluate pairs of perturbations \epsilon, −\epsilon
                self.agent.set_flat_weights(flat_weights + perturbation)
                episode_reward_pos, episode_step_pos = self._play_one_episode(
                    add_noise=True)

                self.agent.set_flat_weights(flat_weights - perturbation)
                episode_reward_neg, episode_step_neg = self._play_one_episode(
                    add_noise=True)

                noise_indices.append(noise_index)
                rewards.append([episode_reward_pos, episode_reward_neg])
                lengths.append([episode_step_pos, episode_step_neg])

        return {
            'noise_indices': noise_indices,
            'noisy_rewards': rewards,
            'noisy_lengths': lengths,
            'eval_rewards': eval_rewards,
            'eval_lengths': eval_lengths
        }

    def get_filter(self, flush_after=False):
        return_filter = self.obs_filter.as_serializable()
        if flush_after:
            self.obs_filter.clear_buffer()
        return return_filter

    def set_filter(self, new_filter):
        self.obs_filter.sync(new_filter)
