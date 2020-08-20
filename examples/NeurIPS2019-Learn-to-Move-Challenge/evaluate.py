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
from parl.utils.window_stat import WindowStat
from parl.remote.client import get_global_client
from parl.utils import machine_info
from shutil import copy2

ACT_DIM = 22
VEL_DIM = 19
OBS_DIM = 98 + VEL_DIM
GAMMA = 0.96
TAU = 0.001
ACTOR_LR = 3e-5
CRITIC_LR = 3e-5


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
        self.model_name = None

    def reset(self):
        self.memory = []


class Evaluator(object):
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

        self.evaluate_result = []

        self.lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.model_queue = queue.Queue()

        self.best_shaping_reward = 0
        self.best_env_reward = 0

        if args.offline_evaluate:
            self.offline_evaluate()
        else:
            t = threading.Thread(target=self.online_evaluate)
            t.start()

        with self.lock:
            while True:
                model_path = self.model_queue.get()
                if not args.offline_evaluate:
                    # online evaluate
                    while not self.model_queue.empty():
                        model_path = self.model_queue.get()
                try:
                    self.agent.restore(model_path)
                    break
                except Exception as e:
                    logger.warn("Agent restore Exception: {} ".format(e))

            self.cur_model = model_path

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

    def offline_evaluate(self):
        ckpt_paths = set([])
        for x in os.listdir(args.saved_models_dir):
            path = os.path.join(args.saved_models_dir, x)
            ckpt_paths.add(path)
        ckpt_paths = list(ckpt_paths)
        steps = [int(x.split('-')[-1]) for x in ckpt_paths]
        sorted_idx = sorted(range(len(steps)), key=lambda k: steps[k])
        ckpt_paths = [ckpt_paths[i] for i in sorted_idx]
        ckpt_paths.reverse()
        logger.info("All checkpoints: {}".format(ckpt_paths))
        for ckpt_path in ckpt_paths:
            self.model_queue.put(ckpt_path)

    def online_evaluate(self):
        last_model_step = None
        while True:
            ckpt_paths = set([])
            for x in os.listdir(args.saved_models_dir):
                path = os.path.join(args.saved_models_dir, x)
                ckpt_paths.add(path)
            if len(ckpt_paths) == 0:
                time.sleep(60)
                continue
            ckpt_paths = list(ckpt_paths)
            steps = [int(x.split('-')[-1]) for x in ckpt_paths]
            sorted_idx = sorted(range(len(steps)), key=lambda k: steps[k])
            ckpt_paths = [ckpt_paths[i] for i in sorted_idx]
            model_step = ckpt_paths[-1].split('-')[-1]
            if model_step != last_model_step:
                logger.info("Adding new checkpoint: :{}".format(
                    ckpt_paths[-1]))
                self.model_queue.put(ckpt_paths[-1])
                last_model_step = model_step
            time.sleep(60)

    def run_remote_sample(self):
        remote_actor = Actor(
            difficulty=args.difficulty,
            vel_penalty_coeff=args.vel_penalty_coeff,
            muscle_penalty_coeff=args.muscle_penalty_coeff,
            penalty_coeff=args.penalty_coeff,
            only_first_target=args.only_first_target)

        actor_state = ActorState()

        while True:
            actor_state.model_name = self.cur_model
            actor_state.reset()

            obs = remote_actor.reset()

            while True:
                if actor_state.model_name != self.cur_model:
                    break

                actor_state.memory.append(
                    TransitionExperience(
                        obs=obs,
                        action=None,
                        reward=None,
                        info=None,
                        timestamp=time.time()))

                action = self.pred_batch(obs)

                obs, reward, done, info = remote_actor.step(action)

                actor_state.memory[-1].reward = reward
                actor_state.memory[-1].info = info
                actor_state.memory[-1].action = action
                if done:
                    self._parse_memory(actor_state)
                    break

    def _parse_memory(self, actor_state):
        mem = actor_state.memory
        n = len(mem)
        episode_shaping_reward = np.sum(
            [exp.info['shaping_reward'] for exp in mem])
        episode_env_reward = np.sum([exp.info['env_reward'] for exp in mem])

        with self.lock:
            if actor_state.model_name == self.cur_model:
                self.evaluate_result.append({
                    'shaping_reward':
                    episode_shaping_reward,
                    'env_reward':
                    episode_env_reward,
                    'episode_length':
                    mem[-1].info['frame_count'],
                    'falldown':
                    not mem[-1].info['timeout'],
                })
                logger.info('{}, finish_cnt: {}'.format(
                    self.cur_model, len(self.evaluate_result)))
                logger.info('{}'.format(self.evaluate_result[-1]))
                if len(self.evaluate_result) >= args.evaluate_times:
                    mean_value = {}
                    for key in self.evaluate_result[0].keys():
                        mean_value[key] = np.mean(
                            [x[key] for x in self.evaluate_result])
                    logger.info('Model: {}, mean_value: {}'.format(
                        self.cur_model, mean_value))

                    eval_num = len(self.evaluate_result)
                    falldown_num = len(
                        [x for x in self.evaluate_result if x['falldown']])
                    falldown_rate = falldown_num / eval_num
                    logger.info('Falldown rate: {}'.format(falldown_rate))
                    for key in self.evaluate_result[0].keys():
                        mean_value[key] = np.mean([
                            x[key] for x in self.evaluate_result
                            if not x['falldown']
                        ])
                    logger.info(
                        'Model: {}, Exclude falldown, mean_value: {}'.format(
                            self.cur_model, mean_value))
                    if mean_value['shaping_reward'] > self.best_shaping_reward:
                        self.best_shaping_reward = mean_value['shaping_reward']
                        copy2(self.cur_model, './model_zoo')
                        logger.info(
                            "[best shaping reward updated:{}] path:{}".format(
                                self.best_shaping_reward, self.cur_model))
                    if mean_value[
                            'env_reward'] > self.best_env_reward and falldown_rate < 0.3:
                        self.best_env_reward = mean_value['env_reward']
                        copy2(self.cur_model, './model_zoo')
                        logger.info(
                            "[best env reward updated:{}] path:{}, falldown rate: {}"
                            .format(self.best_env_reward, self.cur_model,
                                    falldown_num / eval_num))

                    self.evaluate_result = []
                    while True:
                        model_path = self.model_queue.get()
                        if not args.offline_evaluate:
                            # online evaluate
                            while not self.model_queue.empty():
                                model_path = self.model_queue.get()
                        try:
                            self.agent.restore(model_path)
                            break
                        except Exception as e:
                            logger.warn(
                                "Agent restore Exception: {} ".format(e))
                    self.cur_model = model_path
            else:
                actor_state.model_name = self.cur_model
        actor_state.reset()

    def pred_batch(self, obs):
        batch_obs = np.expand_dims(obs, axis=0)
        with self.model_lock:
            action = self.agent.predict(batch_obs.astype('float32'))

        action = np.squeeze(action, axis=0)
        return action


if __name__ == '__main__':
    from evaluate_args import get_args
    args = get_args()
    if args.logdir is not None:
        logger.set_dir(args.logdir)

    evaluate = Evaluator(args)
