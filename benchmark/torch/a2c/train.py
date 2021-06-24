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

import torch
import os
import gym
import parl
import time
import numpy as np

from collections import defaultdict
from parl.env.atari_wrappers import wrap_deepmind
from parl.utils.window_stat import WindowStat
from parl.utils.time_stat import TimeStat
from parl.utils import machine_info
from parl.utils import logger, get_gpu_count, summary
from parl.algorithms import A2C

from atari_model import ActorCritic
from atari_agent import Agent
from actor import Actor

import time
import argparse
import random


class Learner(object):
    def __init__(self, config):

        self.device = torch.device("cuda" if torch.cuda.
                                   is_available() else "cpu")
        self.config = config
        env = gym.make(config['env_name'])
        env = wrap_deepmind(env, dim=config['env_dim'], obs_format='NCHW')
        obs_shape = env.observation_space.shape
        act_dim = env.action_space.n
        self.config['obs_shape'] = obs_shape
        self.config['act_dim'] = act_dim

        model = ActorCritic(act_dim)
        model.to(self.device)
        algorithm = A2C(model, config)
        self.agent = Agent(algorithm, config)

        #========== Learner ==========
        self.total_loss_stat = WindowStat(100)
        self.pi_loss_stat = WindowStat(100)
        self.vf_loss_stat = WindowStat(100)
        self.entropy_stat = WindowStat(100)
        self.lr = None
        self.entropy_coeff = None

        self.learn_time_stat = TimeStat(100)
        self.start_time = None

        #========== Remote Actor ===========
        self.sample_total_steps = 0

        self.create_actors()

    def create_actors(self):
        parl.connect(self.config['master_address'])
        self.remote_actors = [
            Actor(self.config) for _ in range(self.config['actor_num'])
        ]
        logger.info('Creating {} remote actors to connect.'.format(
            self.config['actor_num']))
        self.start_time = time.time()

    def step(self):
        """
        1.setting latest_params to each actor model
        2.getting the sample data from all the actors synchronizely
        3.traing the model with the sample data and the params is upgraded, and goto step 1
        """

        latest_params = self.agent.get_weights()
        # setting the actor to the latest_params
        for remote_actor in self.remote_actors:
            remote_actor.set_weights(latest_params)

        train_batch = defaultdict(list)
        # get the total train data of all the actors.
        future_object_ids = [
            remote_actor.sample() for remote_actor in self.remote_actors
        ]
        sample_datas = [
            future_object.get() for future_object in future_object_ids
        ]
        for sample_data in sample_datas:
            for key, value in sample_data.items():
                train_batch[key].append(value)
            self.sample_total_steps += len(sample_data['obs'])

        for key, value in train_batch.items():
            train_batch[key] = np.concatenate(value)
        with self.learn_time_stat:
            total_loss, pi_loss, vf_loss, entropy, lr, entropy_coeff = self.agent.learn(
                obs=train_batch['obs'],
                actions=train_batch['actions'],
                advantages=train_batch['advantages'],
                target_values=train_batch['target_values'],
            )

        self.total_loss_stat.add(total_loss)
        self.pi_loss_stat.add(pi_loss)
        self.vf_loss_stat.add(vf_loss)
        self.entropy_stat.add(entropy)
        self.lr = lr
        self.entropy_coeff = entropy_coeff

    def log_metrics(self):
        """ Log metrics of learner and actors
        """
        if self.start_time is None:
            return

        metrics = []
        # get the total metrics data
        future_object_ids = [
            remote_actor.get_metrics() for remote_actor in self.remote_actors
        ]
        metrics = [future_object.get() for future_object in future_object_ids]

        # if the metric of all the metrics are empty, return nothing.
        total_length = sum(len(metric) for metric in metrics)
        if not total_length:
            return

        episode_rewards, episode_steps = [], []
        for x in metrics:
            episode_rewards.extend(x['episode_rewards'])
            episode_steps.extend(x['episode_steps'])
        max_episode_rewards, mean_episode_rewards, min_episode_rewards, \
                max_episode_steps, mean_episode_steps, min_episode_steps =\
                None, None, None, None, None, None
        if episode_rewards:
            mean_episode_rewards = np.mean(np.array(episode_rewards).flatten())
            max_episode_rewards = np.max(np.array(episode_rewards).flatten())
            min_episode_rewards = np.min(np.array(episode_rewards).flatten())

            mean_episode_steps = np.mean(np.array(episode_steps).flatten())
            max_episode_steps = np.max(np.array(episode_steps).flatten())
            min_episode_steps = np.min(np.array(episode_steps).flatten())

        metric = {
            'Sample steps': self.sample_total_steps,
            'max_episode_rewards': max_episode_rewards,
            'mean_episode_rewards': mean_episode_rewards,
            'min_episode_rewards': min_episode_rewards,
            'max_episode_steps': max_episode_steps,
            'mean_episode_steps': mean_episode_steps,
            'min_episode_steps': min_episode_steps,
            'total_loss': self.total_loss_stat.mean,
            'pi_loss': self.pi_loss_stat.mean,
            'vf_loss': self.vf_loss_stat.mean,
            'entropy': self.entropy_stat.mean,
            'learn_time_s': self.learn_time_stat.mean,
            'elapsed_time_s': int(time.time() - self.start_time),
            'lr': self.lr,
            'entropy_coeff': self.entropy_coeff,
        }

        for key, value in metric.items():
            if value is not None:
                summary.add_scalar(key, value, self.sample_total_steps)

        logger.info(metric)

    def should_stop(self):
        return self.sample_total_steps >= self.config['max_sample_steps']


if __name__ == '__main__':
    from a2c_config import config
    learner = Learner(config)
    assert config['log_metrics_interval_s'] > 0

    while not learner.should_stop():
        start = time.time()
        while time.time() - start < config['log_metrics_interval_s']:
            learner.step()
        learner.log_metrics()
