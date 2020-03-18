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
import os
import parl
import queue
import six
import time
import threading

from actor import Actor
from collections import defaultdict
from env_wrapper import ObsProcessWrapper, ActionProcessWrapper
from parl.utils import logger, get_gpu_count, tensorboard, machine_info
from parl.utils.scheduler import PiecewiseScheduler
from parl.utils.time_stat import TimeStat
from parl.utils.window_stat import WindowStat
from rlschool import LiftSim
from lift_model import LiftModel
from lift_agent import LiftAgent


class Learner(object):
    def __init__(self, config):
        self.config = config

        #=========== Create Agent ==========
        env = LiftSim()
        env = ActionProcessWrapper(env)
        env = ObsProcessWrapper(env)

        obs_dim = env.obs_dim
        act_dim = env.act_dim
        self.config['obs_dim'] = obs_dim

        model = LiftModel(act_dim)
        algorithm = parl.algorithms.A3C(
            model, vf_loss_coeff=config['vf_loss_coeff'])
        self.agent = LiftAgent(algorithm, config)

        if machine_info.is_gpu_available():
            assert get_gpu_count() == 1, 'Only support training in single GPU,\
                    Please set environment variable: `export CUDA_VISIBLE_DEVICES=[GPU_ID_TO_USE]` .'

        #========== Learner ==========

        self.entropy_stat = WindowStat(100)
        self.target_values = None

        self.learn_time_stat = TimeStat(100)
        self.start_time = None

        #========== Remote Actor ===========
        self.remote_count = 0
        self.sample_data_queue = queue.Queue()

        self.remote_metrics_queue = queue.Queue()
        self.sample_total_steps = 0

        self.params_queues = []
        self.create_actors()

        self.log_steps = 0

    def create_actors(self):
        """ Connect to the cluster and start sampling of the remote actor.
        """
        parl.connect(self.config['master_address'])

        logger.info('Waiting for {} remote actors to connect.'.format(
            self.config['actor_num']))

        for i in six.moves.range(self.config['actor_num']):
            params_queue = queue.Queue()
            self.params_queues.append(params_queue)

            self.remote_count += 1
            logger.info('Remote actor count: {}'.format(self.remote_count))

            remote_thread = threading.Thread(
                target=self.run_remote_sample, args=(params_queue, ))
            remote_thread.setDaemon(True)
            remote_thread.start()

        self.start_time = time.time()

    def run_remote_sample(self, params_queue):
        """ Sample data from remote actor and update parameters of remote actor.
        """
        remote_actor = Actor(self.config)

        cnt = 0
        while True:
            latest_params = params_queue.get()
            remote_actor.set_weights(latest_params)
            batch = remote_actor.sample()

            self.sample_data_queue.put(batch)

            cnt += 1
            if cnt % self.config['get_remote_metrics_interval'] == 0:
                metrics = remote_actor.get_metrics()
                if metrics:
                    self.remote_metrics_queue.put(metrics)

    def step(self):
        """
        1. kick off all actors to synchronize parameters and sample data;
        2. collect sample data of all actors;
        3. update parameters.
        """
        latest_params = self.agent.get_weights()
        for params_queue in self.params_queues:
            params_queue.put(latest_params)

        train_batch = defaultdict(list)
        for i in range(self.config['actor_num']):
            sample_data = self.sample_data_queue.get()
            for key, value in sample_data.items():
                train_batch[key].append(value)

            self.sample_total_steps += sample_data['obs'].shape[0]

        for key, value in train_batch.items():
            train_batch[key] = np.concatenate(value)

        with self.learn_time_stat:
            total_loss, pi_loss, vf_loss, entropy, lr, entropy_coeff = self.agent.learn(
                obs_np=train_batch['obs'],
                actions_np=train_batch['actions'],
                advantages_np=train_batch['advantages'],
                target_values_np=train_batch['target_values'])

        self.entropy_stat.add(entropy)
        self.target_values = np.mean(train_batch['target_values'])

        tensorboard.add_scalar('model/entropy', entropy,
                               self.sample_total_steps)
        tensorboard.add_scalar('model/q_value', self.target_values,
                               self.sample_total_steps)

    def log_metrics(self):
        """ Log metrics of learner and actors
        """
        if self.start_time is None:
            return

        metrics = []
        while True:
            try:
                metric = self.remote_metrics_queue.get_nowait()
                metrics.append(metric)
            except queue.Empty:
                break

        env_reward_1h, env_reward_24h = [], []
        for x in metrics:
            env_reward_1h.extend(x['env_reward_1h'])
            env_reward_24h.extend(x['env_reward_24h'])
        env_reward_1h = [x for x in env_reward_1h if x is not None]
        env_reward_24h = [x for x in env_reward_24h if x is not None]

        mean_reward_1h, mean_reward_24h = None, None
        if env_reward_1h:
            mean_reward_1h = np.mean(np.array(env_reward_1h).flatten())
            tensorboard.add_scalar('performance/env_rewards_1h',
                                   mean_reward_1h, self.sample_total_steps)
        if env_reward_24h:
            mean_reward_24h = np.mean(np.array(env_reward_24h).flatten())
            tensorboard.add_scalar('performance/env_rewards_24h',
                                   mean_reward_24h, self.sample_total_steps)

        metric = {
            'Sample steps': self.sample_total_steps,
            'env_reward_1h': mean_reward_1h,
            'env_reward_24h': mean_reward_24h,
            'target_values': self.target_values,
            'entropy': self.entropy_stat.mean,
            'learn_time_s': self.learn_time_stat.mean,
            'elapsed_time_s': int(time.time() - self.start_time),
        }
        logger.info(metric)

        self.log_steps += 1
        save_interval_step = 7200 // max(1,
                                         self.config['log_metrics_interval_s'])
        if self.log_steps % save_interval_step == 0:
            self.save_model()  # save model every 2h

    def should_stop(self):
        return self.sample_total_steps >= self.config['max_sample_steps']

    def save_model(self):
        time_str = time.strftime(".%Y%m%d_%H%M%S", time.localtime())
        self.agent.save(os.path.join('saved_models', 'model.ckpt' + time_str))


if __name__ == '__main__':
    from a2c_config import config

    learner = Learner(config)
    assert config['log_metrics_interval_s'] > 0

    while not learner.should_stop():
        start = time.time()
        while time.time() - start < config['log_metrics_interval_s']:
            learner.step()
        learner.log_metrics()
