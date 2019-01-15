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

import csv
import numpy as np
import tensorflow as tf
import os
import six


class Summary(object):
    """Logging in tensorboard without tensorflow ops.

    Simple example on how to log scalars and images to tensorboard without tensor ops.
    License: Copyleft

    __author__ = "Michael Gygli"
    """

    def __init__(self, logdir):
        """Creates a summary writer logging to logdir."""
        self.writer = tf.summary.FileWriter(logdir)

    def log_scalar(self, tag, value, step):
        """Log a scalar variable.
        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        step : int
            training iteration
        """
        summary = tf.Summary(
            value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)
        self.writer.flush()


class StatCounter(object):
    """ A simple counter"""

    def __init__(self, max_size=50):
        self.reset()
        self.max_size = max_size

    def feed(self, v):
        """
        Args:
            v(float or np.ndarray): has to be the same shape between calls.
        """
        self._values.append(v)
        if len(self._values) > self.max_size:
            self._values = self._values[-self.max_size:]

    def reset(self):
        self._values = []

    @property
    def count(self):
        return len(self._values)

    @property
    def mean(self):
        assert len(self._values)
        return np.mean(self._values)

    @property
    def sum(self):
        assert len(self._values)
        return np.sum(self._values)

    @property
    def max(self):
        assert len(self._values)
        return max(self._values)

    @property
    def min(self):
        assert len(self._values)
        return min(self._values)

    @property
    def success_rate(self):
        count = 0
        for v in self._values:
            if v > 35.0:
                count += 1
        return float(count) / len(self._values)


def calc_indicators(mem):
    START_STEPS = 15
    n = len(mem)
    episode_shaping_reward = np.sum(
        [exp.info['shaping_reward'] for exp in mem])
    episode_r2_reward = np.sum([
        exp.info['r2_reward'] for exp in mem if exp.info['frame_count'] <= 1000
    ])
    x_offset_reward = np.sum(exp.info['x_offset_reward'] for exp in mem)

    episode_length = mem[-1].info['frame_count']

    scalar_vel = np.mean([exp.info['scalar_vel'] for exp in mem])
    action_l2_penalty = np.mean(
        [exp.info['mean_action_l2_penalty'] for exp in mem])

    start_loss = 10 * START_STEPS * 4 - np.sum([exp.reward
                                                for exp in mem][:START_STEPS])
    all_start_loss = 0
    for i in range(n):
        if not mem[i].info['target_changed']:
            frame_count = 4
            if i - 1 >= 0:
                frame_count = mem[i].info['frame_count'] - mem[
                    i - 1].info['frame_count']
            all_start_loss += 10.0 * frame_count - mem[i].reward
        else:
            break
    start_other_loss = all_start_loss - start_loss
    first_change_loss = 0
    second_change_loss = 0
    third_change_loss = 0
    first_change_other_loss = 0
    second_change_other_loss = 0
    third_change_other_loss = 0
    first_stage_vel = 0.0
    second_stage_vel = 0.0
    third_stage_vel = 0.0
    other_loss = 0

    change_loss = []
    all_change_loss = []
    change_vel = []

    for i in range(n - 1):
        if mem[i].info['target_changed']:
            change_loss.append(0.0)
            all_change_loss.append(0.0)
            change_vel.append([])
            for j in range(START_STEPS):
                idx = i + 1 + j
                if idx >= n or mem[idx].info['target_changed']:
                    break
                frame_count = 4
                if idx - 1 >= 0:
                    frame_count = mem[idx].info['frame_count'] - mem[
                        idx - 1].info['frame_count']
                change_loss[-1] += 10.0 * frame_count - mem[idx].reward
            for j in range(n - i - 1):
                idx = i + 1 + j
                if idx >= n or mem[idx].info['target_changed']:
                    break
                if idx - 1 >= 0:
                    frame_count = mem[idx].info['frame_count'] - mem[
                        idx - 1].info['frame_count']
                all_change_loss[-1] += 10.0 * frame_count - mem[idx].reward
                change_vel[-1].append(mem[idx].info['scalar_vel'])

    if len(change_loss) >= 1:
        first_change_loss = change_loss[0]
        first_change_other_loss = all_change_loss[0] - change_loss[0]
    if len(change_loss) >= 2:
        second_change_loss = change_loss[1]
        second_change_other_loss = all_change_loss[1] - change_loss[1]
    if len(change_loss) >= 3:
        third_change_loss = change_loss[2]
        third_change_other_loss = all_change_loss[2] - change_loss[2]
    other_loss = 10 * mem[-1].info[
        'frame_count'] - start_loss - first_change_loss - second_change_loss - third_change_loss - episode_r2_reward
    if len(change_vel) >= 1:
        first_stage_vel = np.mean(change_vel[0])
    if len(change_vel) >= 2:
        second_stage_vel = np.mean(change_vel[1])
    if len(change_vel) >= 3:
        third_stage_vel = np.mean(change_vel[2])

    indicators_dict = {
        'episode_shaping_reward': episode_shaping_reward,
        'episode_r2_reward': episode_r2_reward,
        'x_offset_reward': x_offset_reward,
        'episode_length': episode_length,
        'scalar_vel': scalar_vel,
        'mean_action_l2_penalty': action_l2_penalty,
        'start_loss': start_loss,
        'first_change_loss': first_change_loss,
        'second_change_loss': second_change_loss,
        'third_change_loss': third_change_loss,
        'start_other_loss': start_other_loss,
        'first_change_other_loss': first_change_other_loss,
        'second_change_other_loss': second_change_other_loss,
        'third_change_other_loss': third_change_other_loss,
        'other_loss': other_loss,
        'first_stage_vel': first_stage_vel,
        'second_stage_vel': second_stage_vel,
        'third_stage_vel': third_stage_vel
    }
    return indicators_dict


class ScalarsManager(object):
    def __init__(self, logdir):
        self.summary = Summary(logdir=logdir)

        self.max_shaping_reward = 0
        self.max_x_offset_reward = 0
        self.max_r2_reward = 0

        self.critic_loss_counter = StatCounter(max_size=500)

        self.r2_reward_counter = StatCounter(max_size=500)
        self.nofall_r2_reward_counter = StatCounter(max_size=500)
        self.falldown_counter100 = StatCounter(max_size=100)

        self.vel_keys = [
            'scalar_vel', 'first_stage_vel', 'second_stage_vel',
            'third_stage_vel'
        ]
        self.vel_counter = {}
        for key in self.vel_keys:
            self.vel_counter[key] = StatCounter(max_size=500)
        self.reward_loss_keys = [
            'start_loss', 'first_change_loss', 'second_change_loss',
            'third_change_loss', 'start_other_loss', 'first_change_other_loss',
            'second_change_other_loss', 'third_change_other_loss', 'other_loss'
        ]
        self.reward_loss_counter = {}
        for key in self.reward_loss_keys:
            self.reward_loss_counter[key] = StatCounter(max_size=500)

    def feed_critic_loss(self, critic_loss):
        self.critic_loss_counter.feed(critic_loss)

    def record(self, record_dict, global_step):
        self.max_shaping_reward = max(self.max_shaping_reward,
                                      record_dict['episode_shaping_reward'])
        self.max_x_offset_reward = max(self.max_x_offset_reward,
                                       record_dict['x_offset_reward'])
        self.max_r2_reward = max(self.max_r2_reward,
                                 record_dict['episode_r2_reward'])

        self.r2_reward_counter.feed(record_dict['episode_r2_reward'])
        if record_dict['episode_length'] >= 1000:  # no falldown
            self.nofall_r2_reward_counter.feed(
                record_dict['episode_r2_reward'])
            self.falldown_counter100.feed(0.0)
        else:
            self.falldown_counter100.feed(1.0)

        for key in self.reward_loss_keys:
            self.reward_loss_counter[key].feed(record_dict[key])
        for key in self.vel_keys:
            self.vel_counter[key].feed(record_dict[key])

        self.summary.log_scalar('performance/falldown_rate',
                                self.falldown_counter100.sum / 100.0,
                                global_step)
        self.summary.log_scalar('performance/max_r2_reward',
                                self.max_r2_reward, global_step)
        self.summary.log_scalar('performance/max_shaping_reward',
                                self.max_shaping_reward, global_step)
        self.summary.log_scalar('performance/max_x_offset_reward',
                                self.max_x_offset_reward, global_step)
        self.summary.log_scalar('performance/episode_r2_reward',
                                record_dict['episode_r2_reward'], global_step)
        self.summary.log_scalar('performance/episode_shaping_reward',
                                record_dict['episode_shaping_reward'],
                                global_step)
        self.summary.log_scalar('performance/x_offset_reward',
                                record_dict['x_offset_reward'], global_step)
        self.summary.log_scalar('performance/episode_length',
                                record_dict['episode_length'], global_step)
        self.summary.log_scalar('performance/mean_action_l2_penalty',
                                record_dict['mean_action_l2_penalty'],
                                global_step)

        self.summary.log_scalar('server/free_client_num',
                                record_dict['free_client_num'], global_step)

        self.summary.log_scalar('model/noiselevel', record_dict['noiselevel'],
                                global_step)
        if self.critic_loss_counter.count > 0:
            mean_critic_loss = self.critic_loss_counter.mean
            self.summary.log_scalar('model/critic_loss', mean_critic_loss,
                                    global_step)

        if self.r2_reward_counter.count > 400:
            mean_r2_reward = self.r2_reward_counter.mean
            self.summary.log_scalar('performance/recent_r2_reward',
                                    mean_r2_reward, global_step)
            mean_nofall_r2_reward = self.nofall_r2_reward_counter.mean
            self.summary.log_scalar('performance/recent_nofall_r2_reward',
                                    mean_nofall_r2_reward, global_step)

            for key in self.vel_keys:
                self.summary.log_scalar('scalar_vel/{}'.format(key),
                                        self.vel_counter[key].mean,
                                        global_step)

            for key in self.reward_loss_keys:
                if 'first' in key:
                    self.summary.log_scalar('1_stage_loss_reward/' + key,
                                            self.reward_loss_counter[key].mean,
                                            global_step)
                elif 'second' in key:
                    self.summary.log_scalar('2_stage_loss_reward/' + key,
                                            self.reward_loss_counter[key].mean,
                                            global_step)
                elif 'third' in key:
                    self.summary.log_scalar('3_stage_loss_reward/' + key,
                                            self.reward_loss_counter[key].mean,
                                            global_step)
                elif 'start' in key:
                    self.summary.log_scalar('0_stage_loss_reward/' + key,
                                            self.reward_loss_counter[key].mean,
                                            global_step)
                else:
                    self.summary.log_scalar('loss_reward/' + key,
                                            self.reward_loss_counter[key].mean,
                                            global_step)
            self.summary.log_scalar(
                '0_stage_loss_reward/stage_loss',
                self.reward_loss_counter['start_loss'].mean +
                self.reward_loss_counter['start_other_loss'].mean, global_step)
            self.summary.log_scalar(
                '1_stage_loss_reward/stage_loss',
                self.reward_loss_counter['first_change_loss'].mean +
                self.reward_loss_counter['first_change_other_loss'].mean,
                global_step)
            self.summary.log_scalar(
                '2_stage_loss_reward/stage_loss',
                self.reward_loss_counter['second_change_loss'].mean +
                self.reward_loss_counter['second_change_other_loss'].mean,
                global_step)
            self.summary.log_scalar(
                '3_stage_loss_reward/stage_loss',
                self.reward_loss_counter['third_change_loss'].mean +
                self.reward_loss_counter['third_change_other_loss'].mean,
                global_step)


class TransitionExperience(object):
    """ A transition of state, or experience"""

    def __init__(self, obs, action, reward, info, **kwargs):
        """ kwargs: whatever other attribute you want to save"""
        self.obs = obs
        self.action = action
        self.reward = reward
        self.info = info
        for k, v in six.iteritems(kwargs):
            setattr(self, k, v)
