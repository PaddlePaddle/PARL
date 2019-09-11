#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import time
import torch
import threading
import numpy as np
from queue import Queue
from utils import get_player
from agent import AtariAgent
from algorithm import DQN
from model import AtariModel
from utils import get_player
from parl.utils import logger, tensorboard


@parl.remote_class
class EvalActor(object):
    def __init__(self, config):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        self.env = get_player(
            config['rom'],
            config['image_size'],
            frame_skip=config['frame_skip'],
            context_len=config['context_len'])
        model = AtariModel(config['context_len'], config['act_dim'],
                           config['algo'])
        algorithm = DQN(
            model,
            act_dim=config['act_dim'],
            gamma=config['gamma'],
            lr=config['lr'],
            algo=config['algo'])
        self.agent = AtariAgent(algorithm, act_dim=config['act_dim'])
        self.eval_nums = config['eval_nums']
        self.weights_queue = Queue(maxsize=1)
        self.result_queue = Queue(maxsize=1)

    def update_weights(self, weights):
        self.weights_queue.put(weights)

    def get_result(self):
        return self.result_queue.get()

    def run_evaluate_episode(self):
        state = self.env.reset()
        total_reward = 0
        t1 = time.time()
        while True:
            pred_Q = self.agent.predict(state)
            action = pred_Q.max(1)[1].item()
            state, reward, isOver, _ = self.env.step(action)
            total_reward += reward
            if isOver:
                logger.info(
                    'Eval one episode with {:.3f}s'.format(time.time() - t1))
                return total_reward

    def evaluate(self):
        while True:
            eval_weights = self.weights_queue.get()
            eval_weights = [
                torch.Tensor(weight).cuda() for weight in eval_weights
            ]
            self.agent.alg.set_weights(eval_weights)
            scores = 0
            for i in range(self.eval_nums):
                scores += self.run_evaluate_episode()
            self.result_queue.put(scores / self.eval_nums)

    def run(self):
        th = threading.Thread(target=self.evaluate)
        th.start()


class EvalModel(object):
    def __init__(self, config):
        self.weights_queue = Queue(maxsize=1)
        self.actors = [EvalActor(config) for _ in range(config['actor_nums'])]

    def run(self):
        for actor in self.actors:
            th = threading.Thread(target=actor.run)
            th.start()

        while True:
            eval_weights, total_steps = self.weights_queue.get()
            results = []
            for actor in self.actors:
                actor.update_weights(eval_weights)
            for actor in self.actors:
                score = actor.get_result()
                results.append(score)
                logger.info(f'get one result {score} for steps {total_steps}')
            tensorboard.add_scalar('dqn/eval', np.mean(results), total_steps)
