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

import time
import gym
import os
import parl
import numpy as np
import utils
from es import ES
from obs_filter import MeanStdFilter
from mujoco_agent import MujocoAgent
from mujoco_model import MujocoModel
from noise import SharedNoiseTable
from parl.utils import logger, summary

from parl.utils.window_stat import WindowStat
from actor import Actor


class Learner(object):
    def __init__(self, config):
        self.config = config

        env = gym.make(self.config['env_name'])
        self.config['obs_dim'] = env.observation_space.shape[0]
        self.config['act_dim'] = env.action_space.shape[0]

        self.obs_filter = MeanStdFilter(self.config['obs_dim'])
        self.noise = SharedNoiseTable(self.config['noise_size'])

        model = MujocoModel(self.config['obs_dim'], self.config['act_dim'])
        algorithm = ES(model)
        self.agent = MujocoAgent(algorithm, self.config)

        self.latest_flat_weights = self.agent.get_flat_weights()
        self.latest_obs_filter = self.obs_filter.as_serializable()

        self.sample_total_episodes = 0
        self.sample_total_steps = 0
        self.train_steps = 0

        self.create_actors()

        self.eval_rewards_stat = WindowStat(self.config['report_window_size'])
        self.eval_lengths_stat = WindowStat(self.config['report_window_size'])

    def create_actors(self):
        """ create actors for parallel training.
        """

        parl.connect(self.config['master_address'])
        self.remote_actors = [
            Actor(self.config) for _ in range(self.config['actor_num'])
        ]
        logger.info('Creating {} remote actors to connect.'.format(
            self.config['actor_num']))
        self.start_time = time.time()

    def step(self):
        """Run a step in ES.

        1. kick off all actors to synchronize weights and sample data;
        2. update parameters of the model based on sampled data.
        3. update global observation filter based on local filters of all actors, and synchronize global
           filter to all actors.
        """
        num_episodes, num_timesteps = 0, 0
        all_results = []

        while num_episodes < self.config['min_episodes_per_batch'] or \
                num_timesteps < self.config['min_steps_per_batch']:
            # setting the lastest to the actors and get the fitness, noise seed sync.
            future_object_ids = [remote_actor.sample(self.latest_flat_weights) \
                for remote_actor in self.remote_actors]
            results = [
                future_object.get() for future_object in future_object_ids
            ]

            for result in results:
                num_episodes += sum(
                    len(pair) for pair in result['noisy_lengths'])
                num_timesteps += sum(
                    sum(pair) for pair in result['noisy_lengths'])
            # each step we need min_episodes_per_batch fitness, but there is no so many actor, so it needs to run
            # many times. The total results arqe in the 'all_results'.
            all_results.extend(results)

        all_noise_indices = []
        all_training_rewards = []
        all_training_lengths = []
        all_eval_rewards = []
        all_eval_lengths = []

        for result in all_results:
            all_eval_rewards.extend(result['eval_rewards'])
            all_eval_lengths.extend(result['eval_lengths'])

            all_noise_indices.extend(result['noise_indices'])
            all_training_rewards.extend(result['noisy_rewards'])
            all_training_lengths.extend(result['noisy_lengths'])

        assert len(all_eval_rewards) == len(all_eval_lengths)
        assert (len(all_noise_indices) == len(all_training_rewards) ==
                len(all_training_lengths))

        self.sample_total_episodes += num_episodes
        self.sample_total_steps += num_timesteps

        eval_rewards = np.array(all_eval_rewards)
        eval_lengths = np.array(all_eval_lengths)
        noise_indices = np.array(all_noise_indices)
        noisy_rewards = np.array(all_training_rewards)
        noisy_lengths = np.array(all_training_lengths)

        # normalize rewards to (-0.5, 0.5), shahe:[batch_size, 2]
        proc_noisy_rewards = utils.compute_centered_ranks(noisy_rewards)
        # noise shape:[batch_size, weight_total_size]
        noises = [
            self.noise.get(index, self.agent.weights_total_size)
            for index in noise_indices
        ]

        # Update the parameters of the model.
        self.agent.learn(proc_noisy_rewards, noises)
        self.train_steps += 1
        self.latest_flat_weights = self.agent.get_flat_weights()

        # Update obs filter to all the actor sync
        self._update_filter()

        # Store the evaluate rewards
        if len(all_eval_rewards) > 0:
            self.eval_rewards_stat.add(np.mean(eval_rewards))
            self.eval_lengths_stat.add(np.mean(eval_lengths))

        metrics = {
            "episodes_this_iter": noisy_lengths.size,
            "sample_total_episodes": self.sample_total_episodes,
            'sample_total_steps': self.sample_total_steps,
            "evaluate_rewards_mean": self.eval_rewards_stat.mean,
            "evaluate_steps_mean": self.eval_lengths_stat.mean,
            "timesteps_this_iter": noisy_lengths.sum(),
        }

        self.log_metrics(metrics)
        return metrics

    def _update_filter(self):
        # Collect filters from all actors and update global filter
        future_object_ids = [remote_actor.get_filter(flush_after=True) \
            for remote_actor in self.remote_actors]
        filters = [future_object.get() for future_object in future_object_ids]
        for actor_filter in filters:
            self.obs_filter.apply_changes(actor_filter)
        # Set_filter of all actors
        self.latest_obs_filter = self.obs_filter.as_serializable()
        [remote_actor.set_filter(self.latest_obs_filter) \
            for remote_actor in self.remote_actors]

    def log_metrics(self, metrics):
        logger.info(metrics)
        for k, v in metrics.items():
            if v is not None:
                summary.add_scalar(k, v, self.train_steps)


if __name__ == '__main__':
    from es_config import config
    logger.info(
        "Before training, it takes a few mimutes to initialize a noise table for exploration"
    )
    learner = Learner(config)
    while learner.train_steps < config['train_steps']:
        learner.step()
