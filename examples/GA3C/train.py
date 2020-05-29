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
import os
import queue
import six
import time
import threading
import parl
from atari_model import AtariModel
from atari_agent import AtariAgent
from collections import defaultdict
from parl.env.atari_wrappers import wrap_deepmind
from parl.utils import logger, get_gpu_count, summary
from parl.utils.scheduler import PiecewiseScheduler
from parl.utils.time_stat import TimeStat
from parl.utils.window_stat import WindowStat
from parl.utils.rl_utils import calc_gae
from parl.utils import machine_info

from actor import Actor


class Learner(object):
    def __init__(self, config):
        self.config = config

        self.sample_data_queue = queue.Queue()
        self.batch_buffer = defaultdict(list)

        #=========== Create Agent ==========
        env = gym.make(config['env_name'])
        env = wrap_deepmind(env, dim=config['env_dim'], obs_format='NCHW')
        obs_shape = env.observation_space.shape
        act_dim = env.action_space.n

        self.config['obs_shape'] = obs_shape
        self.config['act_dim'] = act_dim

        model = AtariModel(act_dim)
        algorithm = parl.algorithms.A3C(
            model, vf_loss_coeff=config['vf_loss_coeff'])
        self.agent = AtariAgent(
            algorithm,
            obs_shape=self.config['obs_shape'],
            predict_thread_num=self.config['predict_thread_num'],
            learn_data_provider=self.learn_data_provider)

        if machine_info.is_gpu_available():
            assert get_gpu_count() == 1, 'Only support training in single GPU,\
                    Please set environment variable: `export CUDA_VISIBLE_DEVICES=[GPU_ID_YOU_WANT_TO_USE]` .'

        else:
            cpu_num = os.environ.get('CPU_NUM')
            assert cpu_num is not None and cpu_num == '1', 'Only support training in single CPU,\
                    Please set environment variable:  `export CPU_NUM=1`.'

        #========== Learner ==========
        self.lr, self.entropy_coeff = None, None
        self.lr_scheduler = PiecewiseScheduler(config['lr_scheduler'])
        self.entropy_coeff_scheduler = PiecewiseScheduler(
            config['entropy_coeff_scheduler'])

        self.total_loss_stat = WindowStat(100)
        self.pi_loss_stat = WindowStat(100)
        self.vf_loss_stat = WindowStat(100)
        self.entropy_stat = WindowStat(100)

        self.learn_time_stat = TimeStat(100)
        self.start_time = None

        # learn thread
        self.learn_thread = threading.Thread(target=self.run_learn)
        self.learn_thread.setDaemon(True)
        self.learn_thread.start()

        self.predict_input_queue = queue.Queue()

        # predict thread
        self.predict_threads = []
        for i in six.moves.range(self.config['predict_thread_num']):
            predict_thread = threading.Thread(
                target=self.run_predict, args=(i, ))
            predict_thread.setDaemon(True)
            predict_thread.start()
            self.predict_threads.append(predict_thread)

        #========== Remote Simulator ===========
        self.remote_count = 0

        self.remote_metrics_queue = queue.Queue()
        self.sample_total_steps = 0

        self.create_actors()

    def learn_data_provider(self):
        """ Data generator for fluid.layers.py_reader
        """
        B = self.config['train_batch_size']
        while True:
            sample_data = self.sample_data_queue.get()
            self.sample_total_steps += len(sample_data['obs'])
            for key in sample_data:
                self.batch_buffer[key].extend(sample_data[key])

            if len(self.batch_buffer['obs']) >= B:
                batch = {}
                for key in self.batch_buffer:
                    batch[key] = np.array(self.batch_buffer[key][:B])

                obs_np = batch['obs'].astype('float32')
                actions_np = batch['actions'].astype('int64')
                advantages_np = batch['advantages'].astype('float32')
                target_values_np = batch['target_values'].astype('float32')

                self.lr = self.lr_scheduler.step()
                self.entropy_coeff = self.entropy_coeff_scheduler.step()

                yield [
                    obs_np, actions_np, advantages_np, target_values_np,
                    self.lr, self.entropy_coeff
                ]

                for key in self.batch_buffer:
                    self.batch_buffer[key] = self.batch_buffer[key][B:]

    def run_predict(self, thread_id):
        """ predict thread
        """
        batch_ident = []
        batch_obs = []
        while True:
            ident, obs = self.predict_input_queue.get()

            batch_ident.append(ident)
            batch_obs.append(obs)
            while len(batch_obs) < self.config['max_predict_batch_size']:
                try:
                    ident, obs = self.predict_input_queue.get_nowait()
                    batch_ident.append(ident)
                    batch_obs.append(obs)
                except queue.Empty:
                    break
            if batch_obs:
                batch_obs = np.array(batch_obs)
                actions, values = self.agent.sample(batch_obs, thread_id)

                for i, ident in enumerate(batch_ident):
                    self.predict_output_queues[ident].put((actions[i],
                                                           values[i]))
                batch_ident = []
                batch_obs = []

    def run_learn(self):
        """ Learn loop
        """
        while True:
            with self.learn_time_stat:
                total_loss, pi_loss, vf_loss, entropy = self.agent.learn()

            self.total_loss_stat.add(total_loss)
            self.pi_loss_stat.add(pi_loss)
            self.vf_loss_stat.add(vf_loss)
            self.entropy_stat.add(entropy)

    def create_actors(self):
        """ Connect to the cluster and start sampling of the remote actor.
        """
        parl.connect(self.config['master_address'])

        logger.info('Waiting for {} remote actors to connect.'.format(
            self.config['actor_num']))

        ident = 0
        self.predict_output_queues = []

        for i in six.moves.range(self.config['actor_num']):

            self.remote_count += 1
            logger.info('Remote simulator count: {}'.format(self.remote_count))
            if self.start_time is None:
                self.start_time = time.time()

            q = queue.Queue()
            self.predict_output_queues.append(q)

            remote_thread = threading.Thread(
                target=self.run_remote_sample, args=(ident, ))
            remote_thread.setDaemon(True)
            remote_thread.start()
            ident += 1

    def run_remote_sample(self, ident):
        """ Interacts with remote simulator.
        """
        remote_actor = Actor(self.config)
        mem = defaultdict(list)

        obs = remote_actor.reset()
        while True:
            self.predict_input_queue.put((ident, obs))
            action, value = self.predict_output_queues[ident].get()

            next_obs, reward, done = remote_actor.step(action)

            mem['obs'].append(obs)
            mem['actions'].append(action)
            mem['rewards'].append(reward)
            mem['values'].append(value)

            if done:
                next_value = 0
                advantages = calc_gae(mem['rewards'], mem['values'],
                                      next_value, self.config['gamma'],
                                      self.config['lambda'])
                target_values = advantages + mem['values']

                self.sample_data_queue.put({
                    'obs': mem['obs'],
                    'actions': mem['actions'],
                    'advantages': advantages,
                    'target_values': target_values
                })

                mem = defaultdict(list)

                next_obs = remote_actor.reset()

            elif len(mem['obs']) == self.config['t_max'] + 1:
                next_value = mem['values'][-1]
                advantages = calc_gae(mem['rewards'][:-1], mem['values'][:-1],
                                      next_value, self.config['gamma'],
                                      self.config['lambda'])
                target_values = advantages + mem['values'][:-1]

                self.sample_data_queue.put({
                    'obs': mem['obs'][:-1],
                    'actions': mem['actions'][:-1],
                    'advantages': advantages,
                    'target_values': target_values
                })

                for key in mem:
                    mem[key] = [mem[key][-1]]

            obs = next_obs

            if done:
                metrics = remote_actor.get_metrics()
                if metrics:
                    self.remote_metrics_queue.put(metrics)

    def log_metrics(self):
        """ Log metrics of learner and simulators
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


if __name__ == '__main__':
    from ga3c_config import config

    learner = Learner(config)
    assert config['log_metrics_interval_s'] > 0

    while True:
        time.sleep(config['log_metrics_interval_s'])

        learner.log_metrics()
