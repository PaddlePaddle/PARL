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
import parl
import queue
import six
import threading
import time
import numpy as np
from actor import Actor
from opensim_model import OpenSimModel
from opensim_agent import OpenSimAgent
from parl.utils import logger, summary, get_gpu_count
from replay_memory import ReplayMemory
from parl.utils.window_stat import WindowStat
from parl.remote.client import get_global_client
from parl.utils import machine_info

ACT_DIM = 22
VEL_DIM = 19
OBS_DIM = 98 + VEL_DIM
GAMMA = 0.96
TAU = 0.001
ACTOR_LR = 3e-5
CRITIC_LR = 3e-5
BATCH_SIZE = 128
NOISE_DECAY = 0.999998


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


class ActorState(object):
    """Maintain incomplete trajectories data of actor."""

    def __init__(self):
        self.memory = []  # list of Experience
        self.ident = np.random.randint(int(1e18))
        self.last_target_changed_steps = 0

    def reset(self):
        self.memory = []
        self.last_target_changed_steps = 0

    def update_last_target_changed(self):
        self.last_target_changed_steps = len(self.memory)


class Learner(object):
    def __init__(self, args):
        if machine_info.is_gpu_available():
            assert get_gpu_count() == 1, 'Only support training in single GPU,\
                    Please set environment variable: `export CUDA_VISIBLE_DEVICES=[GPU_ID_TO_USE]` .'

        else:
            cpu_num = os.environ.get('CPU_NUM')
            assert cpu_num is not None and cpu_num == '1', 'Only support training in single CPU,\
                    Please set environment variable:  `export CPU_NUM=1`.'

        model = OpenSimModel(OBS_DIM, VEL_DIM, ACT_DIM)
        algorithm = parl.algorithms.DDPG(
            model,
            gamma=GAMMA,
            tau=TAU,
            actor_lr=ACTOR_LR,
            critic_lr=CRITIC_LR)
        self.agent = OpenSimAgent(algorithm, OBS_DIM, ACT_DIM)

        self.rpm = ReplayMemory(args.rpm_size, OBS_DIM, ACT_DIM)

        if args.restore_rpm_path is not None:
            self.rpm.load(args.restore_rpm_path)
        if args.restore_model_path is not None:
            self.restore(args.restore_model_path)

        # add lock between training and predicting
        self.model_lock = threading.Lock()

        # add lock when appending data to rpm or writing scalars to summary
        self.memory_lock = threading.Lock()

        self.ready_actor_queue = queue.Queue()

        self.total_steps = 0
        self.noiselevel = 0.5

        self.critic_loss_stat = WindowStat(500)
        self.env_reward_stat = WindowStat(500)
        self.shaping_reward_stat = WindowStat(500)
        self.max_env_reward = 0

        # thread to keep training
        learn_thread = threading.Thread(target=self.keep_training)
        learn_thread.setDaemon(True)
        learn_thread.start()

        self.create_actors()

    def create_actors(self):
        """Connect to the cluster and start sampling of the remote actor.
        """
        parl.connect(args.cluster_address, ['official_obs_scaler.npz'])

        for i in range(args.actor_num):
            logger.info('Remote actor count: {}'.format(i + 1))

            remote_thread = threading.Thread(target=self.run_remote_sample)
            remote_thread.setDaemon(True)
            remote_thread.start()

        # There is a memory-leak problem in osim-rl package.
        # So we will dynamically add actors when remote actors killed due to excessive memory usage.
        time.sleep(10 * 60)
        parl_client = get_global_client()
        while True:
            if parl_client.actor_num < args.actor_num:
                logger.info(
                    'Dynamic adding acotr, current actor num:{}'.format(
                        parl_client.actor_num))
                remote_thread = threading.Thread(target=self.run_remote_sample)
                remote_thread.setDaemon(True)
                remote_thread.start()
            time.sleep(5)

    def _new_ready_actor(self):
        """ 

        The actor is ready to start new episode,
        but blocking until training thread call actor_ready_event.set()
        """
        actor_ready_event = threading.Event()
        self.ready_actor_queue.put(actor_ready_event)
        logger.info(
            "[new_avaliabe_actor] approximate size of ready actors:{}".format(
                self.ready_actor_queue.qsize()))
        actor_ready_event.wait()

    def run_remote_sample(self):
        remote_actor = Actor(
            difficulty=args.difficulty,
            vel_penalty_coeff=args.vel_penalty_coeff,
            muscle_penalty_coeff=args.muscle_penalty_coeff,
            penalty_coeff=args.penalty_coeff,
            only_first_target=args.only_first_target)

        actor_state = ActorState()

        while True:
            obs = remote_actor.reset()
            actor_state.reset()

            while True:
                actor_state.memory.append(
                    TransitionExperience(
                        obs=obs,
                        action=None,
                        reward=None,
                        info=None,
                        timestamp=time.time()))

                action = self.pred_batch(obs)

                # For each target, decay noise as the steps increase.
                step = len(
                    actor_state.memory) - actor_state.last_target_changed_steps
                current_noise = self.noiselevel * (0.98**(step - 1))

                noise = np.zeros((ACT_DIM, ), dtype=np.float32)
                if actor_state.ident % 3 == 0:
                    if step % 5 == 0:
                        noise = np.random.randn(ACT_DIM) * current_noise
                elif actor_state.ident % 3 == 1:
                    if step % 5 == 0:
                        noise = np.random.randn(ACT_DIM) * current_noise * 2
                action += noise

                action = np.clip(action, -1, 1)

                obs, reward, done, info = remote_actor.step(action)

                reward_scale = (1 - GAMMA)
                info['shaping_reward'] *= reward_scale

                actor_state.memory[-1].reward = reward
                actor_state.memory[-1].info = info
                actor_state.memory[-1].action = action

                if 'target_changed' in info and info['target_changed']:
                    actor_state.update_last_target_changed()

                if done:
                    self._parse_memory(actor_state, last_obs=obs)
                    break

            self._new_ready_actor()

    def _parse_memory(self, actor_state, last_obs):
        mem = actor_state.memory
        n = len(mem)

        episode_shaping_reward = np.sum(
            [exp.info['shaping_reward'] for exp in mem])
        episode_env_reward = np.sum([exp.info['env_reward'] for exp in mem])
        episode_time = time.time() - mem[0].timestamp

        episode_rpm = []
        for i in range(n - 1):
            episode_rpm.append([
                mem[i].obs, mem[i].action, mem[i].info['shaping_reward'],
                mem[i + 1].obs, False
            ])
        episode_rpm.append([
            mem[-1].obs, mem[-1].action, mem[-1].info['shaping_reward'],
            last_obs, not mem[-1].info['timeout']
        ])

        with self.memory_lock:
            self.total_steps += n
            self.add_episode_rpm(episode_rpm)

            if actor_state.ident % 3 == 2:  # trajectory without noise
                self.env_reward_stat.add(episode_env_reward)
                self.shaping_reward_stat.add(episode_shaping_reward)
                self.max_env_reward = max(self.max_env_reward,
                                          episode_env_reward)

                if self.env_reward_stat.count > 500:
                    summary.add_scalar('recent_env_reward',
                                       self.env_reward_stat.mean,
                                       self.total_steps)
                    summary.add_scalar('recent_shaping_reward',
                                       self.shaping_reward_stat.mean,
                                       self.total_steps)
                if self.critic_loss_stat.count > 500:
                    summary.add_scalar('recent_critic_loss',
                                       self.critic_loss_stat.mean,
                                       self.total_steps)
                summary.add_scalar('episode_length', n, self.total_steps)
                summary.add_scalar('max_env_reward', self.max_env_reward,
                                   self.total_steps)
                summary.add_scalar('ready_actor_num',
                                   self.ready_actor_queue.qsize(),
                                   self.total_steps)
                summary.add_scalar('episode_time', episode_time,
                                   self.total_steps)

            self.noiselevel = self.noiselevel * NOISE_DECAY

    def learn(self):
        start_time = time.time()

        for T in range(args.train_times):
            [states, actions, rewards, new_states,
             dones] = self.rpm.sample_batch(BATCH_SIZE)
            with self.model_lock:
                critic_loss = self.agent.learn(states, actions, rewards,
                                               new_states, dones)
            self.critic_loss_stat.add(critic_loss)
        logger.info(
            "[learn] time consuming:{}".format(time.time() - start_time))

    def keep_training(self):
        episode_count = 1000000
        for T in range(episode_count):
            if self.rpm.size() > BATCH_SIZE * args.warm_start_batchs:
                self.learn()
                logger.info(
                    "[keep_training/{}] trying to acq a new env".format(T))

            # Keep training and predicting balance
            # After training, wait for a ready actor, and make the actor start new episode
            ready_actor_event = self.ready_actor_queue.get()
            ready_actor_event.set()

            if np.mod(T, 100) == 0:
                logger.info("saving models")
                self.save(T)
            if np.mod(T, 10000) == 0:
                logger.info("saving rpm")
                self.save_rpm()

    def save_rpm(self):
        save_path = os.path.join(logger.get_dir(), "rpm.npz")
        self.rpm.save(save_path)

    def save(self, T):
        save_path = os.path.join(
            logger.get_dir(), 'model_every_100_episodes/episodes-{}'.format(T))
        self.agent.save(save_path)

    def restore(self, model_path):
        logger.info('restore model from {}'.format(model_path))
        self.agent.restore(model_path)

    def add_episode_rpm(self, episode_rpm):
        for x in episode_rpm:
            self.rpm.append(
                obs=x[0], act=x[1], reward=x[2], next_obs=x[3], terminal=x[4])

    def pred_batch(self, obs):
        batch_obs = np.expand_dims(obs, axis=0)

        with self.model_lock:
            action = self.agent.predict(batch_obs.astype('float32'))

        action = np.squeeze(action, axis=0)
        return action


if __name__ == '__main__':
    from train_args import get_args
    args = get_args()
    if args.logdir is not None:
        logger.set_dir(args.logdir)

    learner = Learner(args)
