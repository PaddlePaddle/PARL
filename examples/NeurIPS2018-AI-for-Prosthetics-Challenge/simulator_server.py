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

import grpc
import json
import numpy as np
import os
import queue
import simulator_pb2
import simulator_pb2_grpc
import six
import time
import threading
from args import get_server_args
from collections import defaultdict
from concurrent import futures
from multi_head_ddpg import MultiHeadDDPG
from opensim_agent import OpenSimAgent
from opensim_model import OpenSimModel
from parl.utils import logger
from replay_memory import ReplayMemory
from utils import calc_indicators, ScalarsManager, TransitionExperience

ACT_DIM = 19
VEL_DIM = 4
OBS_DIM = 185 + VEL_DIM
GAMMA = 0.96
TAU = 0.001
ACTOR_LR = 3e-5
CRITIC_LR = 3e-5
TRAIN_TIMES = 100
BATCH_SIZE = 128
NOISE_DECAY = 0.999998
_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class SimulatorServer(simulator_pb2_grpc.SimulatorServicer):
    class ClientState(object):
        def __init__(self):
            self.memory = []  # list of Experience
            self.ident = None
            self.model_idx = np.random.randint(args.ensemble_num)
            self.last_target_changed = 0
            self.target_change_times = 0

        def reset(self):
            self.last_target_changed = 0
            self.memory = []
            self.model_idx = np.random.randint(args.ensemble_num)
            self.target_change_times = 0

        def update_last_target_changed(self):
            self.last_target_changed = len(self.memory)

    def __init__(self):
        self.rpm = ReplayMemory(int(2e6), OBS_DIM, ACT_DIM)

        # Need acquire lock when model learning or predicting
        self.locks = []
        for i in range(args.ensemble_num):
            self.locks.append(threading.Lock())

        models = []
        for i in range(args.ensemble_num):
            models.append(OpenSimModel(OBS_DIM, VEL_DIM, ACT_DIM, model_id=i))

        hyperparas = {
            'gamma': GAMMA,
            'tau': TAU,
            'ensemble_num': args.ensemble_num
        }
        alg = MultiHeadDDPG(models, hyperparas)

        self.agent = OpenSimAgent(alg, OBS_DIM, ACT_DIM, args.ensemble_num)

        self.scalars_manager = ScalarsManager(logger.get_dir())

        # add lock when appending data to rpm or writing scalars to tensorboard
        self.MEMORY_LOCK = threading.Lock()

        self.clients = defaultdict(self.ClientState)

        self.ready_client_queue = queue.Queue()

        self.noiselevel = 0.5
        self.global_step = 0

        # thread to keep training
        t = threading.Thread(target=self.keep_training)
        t.start()

    def _new_ready_client(self):
        """ The client is ready to start new episode,
        but blocking until training thread call client_ready_event.set()
    """
        client_ready_event = threading.Event()
        self.ready_client_queue.put(client_ready_event)
        logger.info(
            "[new_ready_client] approximate size of ready clients:{}".format(
                self.ready_client_queue.qsize()))
        client_ready_event.wait()

    def Send(self, request, context):
        """ Implement Send function in SimulatorServicer
        Everytime a request comming, will create a new thread to handle
    """
        ident, obs, reward, done, info = request.id, request.observation, request.reward, request.done, request.info
        client = self.clients[ident]
        info = json.loads(info)

        if 'first' in info:
            # Waiting training thread to allow start new episode
            self._new_ready_client()

        obs = np.array(obs, dtype=np.float32)
        self._process_msg(ident, obs, reward, done, info)

        if done:
            # Waiting training thread to allow start new episode
            self._new_ready_client()

        action = self.pred_batch(obs, client.model_idx)
        step = len(client.memory) - client.last_target_changed

        # whether to add noise depends on the ensemble_num
        if args.ensemble_num == 1:
            current_noise = self.noiselevel * (0.98**(step - 1))
            noise = np.zeros((ACT_DIM, ), dtype=np.float32)
            if ident % 3 == 0:
                if step % 5 == 0:
                    noise = np.random.randn(ACT_DIM) * current_noise
            elif ident % 3 == 1:
                if step % 5 == 0:
                    noise = np.random.randn(ACT_DIM) * current_noise * 2
            action += noise
        action = np.clip(action, -1, 1)
        client.memory[-1].action = action
        extra_info = {}
        return simulator_pb2.Reply(action=action, extra=json.dumps(extra_info))

    def _process_msg(self, ident, obs, reward, done, info):
        client = self.clients[ident]
        reward_scale = (1 - GAMMA)
        info['shaping_reward'] *= reward_scale
        if len(client.memory) > 0:
            client.memory[-1].reward = reward
            info['target_change_times'] = client.target_change_times
            client.memory[-1].info = info
            if info['target_changed']:
                client.target_change_times = min(
                    client.target_change_times + 1, 3)
                # re-sample model_idx after target was changed
                client.model_idx = np.random.randint(args.ensemble_num)
            if done:
                assert 'last_obs' in info
                self._parse_memory(client, ident, info['last_obs'])
        client.memory.append(
            TransitionExperience(obs=obs, action=None, reward=None, info=None))
        if 'target_changed' in info and info['target_changed']:
            client.update_last_target_changed()
        return False

    def _parse_memory(self, client, ident, last_obs):
        mem = client.memory
        n = len(mem)

        # debug info
        if ident == 1:
            for i, exp in enumerate(mem):
                logger.info(
                    "[step:{}] obs:{} action:{} reward:{} shaping_reward:{}".
                    format(i, np.sum(mem[i].obs), np.sum(mem[i].action),
                           mem[i].reward, mem[i].info['shaping_reward']))

        episode_rpm = []
        for i in range(n - 1):
            if not mem[i].info['target_changed']:
                episode_rpm.append([
                    mem[i].obs, mem[i].action, mem[i].info['shaping_reward'],
                    mem[i + 1].obs, False, mem[i].info['target_change_times']
                ])
        if not mem[-1].info['target_changed']:
            episode_rpm.append([
                mem[-1].obs, mem[-1].action, mem[-1].info['shaping_reward'],
                last_obs, not mem[-1].info['timeout'],
                mem[i].info['target_change_times']
            ])

        indicators_dict = calc_indicators(mem)
        indicators_dict['free_client_num'] = self.ready_client_queue.qsize()
        indicators_dict['noiselevel'] = self.noiselevel

        with self.MEMORY_LOCK:
            self.add_episode_rpm(episode_rpm)
            self.scalars_manager.record(indicators_dict, self.global_step)
            self.global_step += 1
            if self.global_step >= 50:
                self.noiselevel = self.noiselevel * NOISE_DECAY

        client.reset()

    def learn(self):
        result_q = queue.Queue()
        th_list = []
        for j in range(args.ensemble_num):
            t = threading.Thread(
                target=self.train_single_model, args=(j, result_q))
            th_list.append(t)
        start_time = time.time()
        for t in th_list:
            t.start()
        for t in th_list:
            t.join()

        logger.info("[learn] {} heads, time consuming:{}".format(
            args.ensemble_num,
            time.time() - start_time))
        for t in th_list:
            result = result_q.get()
            for critic_loss in result:
                self.scalars_manager.feed_critic_loss(critic_loss)

    def train_single_model(self, model_idx, result_q):
        logger.info("[train_single_model] model_idx:{}".format(model_idx))
        critic_loss_list = []
        lock = self.locks[model_idx]
        memory = self.rpm

        actor_lr = ACTOR_LR * (1.0 - 0.05 * model_idx)
        critic_lr = CRITIC_LR * (1.0 + 0.1 * model_idx)

        for T in range(TRAIN_TIMES):
            [states, actions, rewards, new_states,
             dones] = memory.sample_batch(BATCH_SIZE)
            lock.acquire()
            critic_loss = self.agent.learn(states, actions, rewards,
                                           new_states, dones, actor_lr,
                                           critic_lr, model_idx)
            lock.release()
            critic_loss_list.append(critic_loss)
        result_q.put(critic_loss_list)

    def keep_training(self):
        episode_count = 1000000
        for T in range(episode_count):
            if self.rpm.size() > BATCH_SIZE * args.warm_start_batchs:
                self.learn()
                logger.info(
                    "[keep_training/{}] trying to acq a new env".format(T))

            # Keep training and predicting balance
            # After training, waiting for a ready client, and set the client start new episode
            ready_client_event = self.ready_client_queue.get()
            ready_client_event.set()

            if np.mod(T, 100) == 0:
                logger.info("saving models")
                self.save(T)
            if np.mod(T, 10000) == 0:
                logger.info("saving rpm")
                self.save_rpm()

    def save_rpm(self):
        save_path = os.path.join(logger.get_dir(), "rpm.npz")
        self.rpm.save(save_path)

    def restore_rpm(self, rpm_dir):
        self.rpm.load(rpm_dir)

    def save(self, T):
        save_path = os.path.join(logger.get_dir(),
                                 'model_every_100_episodes/step-{}'.format(T))
        self.agent.save_params(save_path)

    def restore(self, model_path, restore_from_one_head):
        logger.info('restore model from {}'.format(model_path))
        self.agent.load_params(model_path, restore_from_one_head)

    def add_episode_rpm(self, episode_rpm):
        for x in episode_rpm:
            self.rpm.append(
                obs=x[0], act=x[1], reward=x[2], next_obs=x[3], terminal=x[4])

    def pred_batch(self, obs, model_idx=None):
        assert model_idx is not None
        batch_obs = np.expand_dims(obs, axis=0)
        self.locks[model_idx].acquire()
        action = self.agent.predict(batch_obs, model_idx)
        self.locks[model_idx].release()
        action = np.squeeze(action, axis=0)
        return action


class SimulatorHandler(threading.Thread):
    def __init__(self, simulator_server):
        threading.Thread.__init__(self)
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=400))
        simulator_pb2_grpc.add_SimulatorServicer_to_server(
            simulator_server, self.server)
        self.server.add_insecure_port('[::]:{}'.format(args.port))

    def run(self):
        self.server.start()
        try:
            while True:
                time.sleep(_ONE_DAY_IN_SECONDS)
        except KeyboardInterrupt:
            self.server.stop(0)


if __name__ == '__main__':
    args = get_server_args()

    if args.logdir is not None:
        logger.set_dir(args.logdir)

    simulator_server = SimulatorServer()

    if args.restore_rpm_path is not None:
        simulator_server.restore_rpm(args.restore_rpm_path)
    if args.restore_model_path is not None:
        simulator_server.restore(args.restore_model_path,
                                 args.restore_from_one_head)

    simulator_hanlder = SimulatorHandler(simulator_server=simulator_server)
    simulator_hanlder.run()
