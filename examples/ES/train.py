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
import os
import parl
import numpy as np
import threading
import utils
from es import ES
from obs_filter import MeanStdFilter
from mujoco_agent import MujocoAgent
from mujoco_model import MujocoModel
from noise import SharedNoiseTable
from parl.utils import logger, summary
from parl.utils.window_stat import WindowStat
from six.moves import queue
from actor import Actor


class Learner(object):
    def __init__(self, config):
        self.config = config

        env = gym.make(self.config['env_name'])
        self.config['obs_dim'] = env.observation_space.shape[0]
        self.config['act_dim'] = env.action_space.shape[0]

        self.obs_filter = MeanStdFilter(self.config['obs_dim'])
        self.noise = SharedNoiseTable(self.config['noise_size'])

        model = MujocoModel(self.config['act_dim'])
        algorithm = ES(model)
        self.agent = MujocoAgent(algorithm, self.config)

        self.latest_flat_weights = self.agent.get_flat_weights()
        self.latest_obs_filter = self.obs_filter.as_serializable()

        self.sample_total_episodes = 0
        self.sample_total_steps = 0

        self.actors_signal_input_queues = []
        self.actors_output_queues = []

        self.create_actors()

        self.eval_rewards_stat = WindowStat(self.config['report_window_size'])
        self.eval_lengths_stat = WindowStat(self.config['report_window_size'])

    def create_actors(self):
        """ create actors for parallel training.
        """

        parl.connect(self.config['master_address'])
        self.remote_count = 0
        for i in range(self.config['actor_num']):
            signal_queue = queue.Queue()
            output_queue = queue.Queue()
            self.actors_signal_input_queues.append(signal_queue)
            self.actors_output_queues.append(output_queue)

            self.remote_count += 1

            remote_thread = threading.Thread(
                target=self.run_remote_sample,
                args=(signal_queue, output_queue))
            remote_thread.setDaemon(True)
            remote_thread.start()

        logger.info('All remote actors are ready, begin to learn.')

    def run_remote_sample(self, signal_queue, output_queue):
        """ Sample data from remote actor or get filters of remote actor. 
        """
        remote_actor = Actor(self.config)
        while True:
            info = signal_queue.get()
            if info['signal'] == 'sample':
                result = remote_actor.sample(self.latest_flat_weights)
                output_queue.put(result)
            elif info['signal'] == 'get_filter':
                actor_filter = remote_actor.get_filter(flush_after=True)
                output_queue.put(actor_filter)
            elif info['signal'] == 'set_filter':
                remote_actor.set_filter(self.latest_obs_filter)
            else:
                raise NotImplementedError

    def step(self):
        """Run a step in ES.

        1. kick off all actors to synchronize weights and sample data;
        2. update parameters of the model based on sampled data.
        3. update global observation filter based on local filters of all actors, and synchronize global 
           filter to all actors.
        """
        num_episodes, num_timesteps = 0, 0
        results = []

        while num_episodes < self.config['min_episodes_per_batch'] or \
                num_timesteps < self.config['min_steps_per_batch']:
            # Send sample signal to all actors
            for q in self.actors_signal_input_queues:
                q.put({'signal': 'sample'})

            # Collect results from all actors
            for q in self.actors_output_queues:
                result = q.get()
                results.append(result)
                # result['noisy_lengths'] is a list of lists, where the inner lists have length 2.
                num_episodes += sum(
                    len(pair) for pair in result['noisy_lengths'])
                num_timesteps += sum(
                    sum(pair) for pair in result['noisy_lengths'])

        all_noise_indices = []
        all_training_rewards = []
        all_training_lengths = []
        all_eval_rewards = []
        all_eval_lengths = []

        for result in results:
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

        # normalize rewards to (-0.5, 0.5)
        proc_noisy_rewards = utils.compute_centered_ranks(noisy_rewards)
        noises = [
            self.noise.get(index, self.agent.weights_total_size)
            for index in noise_indices
        ]

        # Update the parameters of the model.
        self.agent.learn(proc_noisy_rewards, noises)
        self.latest_flat_weights = self.agent.get_flat_weights()

        # Update obs filter
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
        # Send get_filter signal to all actors
        for q in self.actors_signal_input_queues:
            q.put({'signal': 'get_filter'})

        filters = []
        # Collect filters from  all actors and update global filter
        for q in self.actors_output_queues:
            actor_filter = q.get()
            self.obs_filter.apply_changes(actor_filter)

        # Send set_filter signal to all actors
        self.latest_obs_filter = self.obs_filter.as_serializable()
        for q in self.actors_signal_input_queues:
            q.put({'signal': 'set_filter'})

    def log_metrics(self, metrics):
        logger.info(metrics)
        for k, v in metrics.items():
            if v is not None:
                summary.add_scalar(k, v, self.sample_total_steps)


if __name__ == '__main__':
    from es_config import config

    logger.info(
        "Before training, it takes a few mimutes to initialize a noise table for exploration"
    )
    learner = Learner(config)

    while True:
        learner.step()
