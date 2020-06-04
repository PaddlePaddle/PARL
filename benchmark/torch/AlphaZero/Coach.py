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
import sys
import threading
import queue
import pickle
from pickle import Pickler, Unpickler
from random import shuffle
from parl.utils import tensorboard

import numpy as np
from tqdm import tqdm

import parl
from parl.utils import logger

from actor import Actor
from utils import split_group, get_test_dataset
from alphazero_agent import create_agent


class Coach():
    """
    This class executes the self-play, learning and evaluating. 
    """

    def __init__(self, game, args):
        self.game = game
        self.args = args

        # neural network of current generation
        self.current_agent = create_agent(self.game)
        # neural network of previous generation
        self.previous_agent = create_agent(self.game)

        # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.trainExamplesHistory = []

        self.remote_actors_signal_queues = []
        self.remote_actors_return_queue = queue.Queue()

        self.test_dataset = get_test_dataset()

    def _run_remote_tasks(self, signal_queue):
        # The remote actor will actually run on the local machine or other machines of xparl cluster
        remote_actor = Actor(self.game, self.args)

        while True:
            # receive running task signal
            # signal: specify task type and task input data (optional)
            signal = signal_queue.get()

            if signal["task"] == "self-play":
                episode_num_each_actor = self.args.numEps // self.args.actors_num
                result = remote_actor.self_play(
                    self.current_agent.get_weights(), episode_num_each_actor)
                self.remote_actors_return_queue.put({"self-play": result})

            elif signal["task"] == "pitting":
                games_num_each_actor = self.args.arenaCompare // self.args.actors_num
                result = remote_actor.pitting(
                    self.previous_agent.get_weights(),
                    self.current_agent.get_weights(), games_num_each_actor)
                self.remote_actors_return_queue.put({"pitting": result})

            elif signal["task"] == "evaluate_test_dataset":
                test_dataset = signal["test_dataset"]
                result = remote_actor.evaluate_test_dataset(
                    self.current_agent.get_weights(), test_dataset)
                self.remote_actors_return_queue.put({
                    "evaluate_test_dataset":
                    result
                })
            else:
                raise NotImplementedError

    def _create_remote_actors(self):
        # connect to xparl cluster to submit jobs
        parl.connect(self.args.master_address)

        for i in range(self.args.actors_num):
            signal_queue = queue.Queue()
            self.remote_actors_signal_queues.append(signal_queue)

            remote_thread = threading.Thread(
                target=self._run_remote_tasks, args=(signal_queue, ))
            remote_thread.setDaemon(True)
            remote_thread.start()

    def learn(self):
        """Each iteration:
        1. Performs numEps episodes of self-play.
        2. Retrains neural network with examples in trainExamplesHistory
           (which has a maximum length of numItersForTrainExamplesHistory).
        3. Evaluates the new neural network with the test dataset.
        4. Pits the new neural network against the old one and accepts it
           only if it wins >= updateThreshold fraction of games.
        """

        # create remote actors to run tasks (self-play/pitting/evaluate_test_dataset) in parallel.
        self._create_remote_actors()

        for iteration in range(1, self.args.numIters + 1):
            logger.info('Starting Iter #{} ...'.format(iteration))

            ####################
            logger.info('Step1: self-play in parallel...')
            iterationTrainExamples = []
            # update weights of remote actors to the latest weights, and ask them to run self-play task
            for signal_queue in self.remote_actors_signal_queues:
                signal_queue.put({"task": "self-play"})
            # wait for all remote actors (a total of self.args.actors_num) to return the self-play results
            for _ in range(self.args.actors_num):
                result = self.remote_actors_return_queue.get()
                iterationTrainExamples.extend(result["self-play"])

            # save the iteration examples to the history
            self.trainExamplesHistory.append(iterationTrainExamples)
            if len(self.trainExamplesHistory
                   ) > self.args.numItersForTrainExamplesHistory:
                logger.warning("Removing the oldest entry in trainExamples.")
                self.trainExamplesHistory.pop(0)
            self.saveTrainExamples(iteration)  # backup history to a file

            ####################
            logger.info('Step2: train neural network...')
            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.current_agent.save(
                os.path.join(self.args.checkpoint, 'temp.pth.tar'))
            self.previous_agent.restore(
                os.path.join(self.args.checkpoint, 'temp.pth.tar'))

            self.current_agent.learn(trainExamples)

            ####################
            logger.info('Step3: evaluate test dataset in parallel...')
            cnt = 0
            # update weights of remote actors to the latest weights, and ask them to evaluate assigned test dataset
            for i, data in enumerate(
                    split_group(
                        self.test_dataset,
                        len(self.test_dataset) // self.args.actors_num)):
                self.remote_actors_signal_queues[i].put({
                    "task":
                    "evaluate_test_dataset",
                    "test_dataset":
                    data
                })
                cnt += len(data)
            perfect_moves_cnt, good_moves_cnt = 0, 0
            # wait for all remote actors (a total of self.args.actors_num) to return the evaluating results
            for _ in range(self.args.actors_num):
                (perfect_moves,
                 good_moves) = self.remote_actors_return_queue.get(
                 )["evaluate_test_dataset"]
                perfect_moves_cnt += perfect_moves
                good_moves_cnt += good_moves
            logger.info('perfect moves rate: {}, good moves rate: {}'.format(
                perfect_moves_cnt / cnt, good_moves_cnt / cnt))
            tensorboard.add_scalar('perfect_moves_rate',
                                   perfect_moves_cnt / cnt, iteration)
            tensorboard.add_scalar('good_moves_rate', good_moves_cnt / cnt,
                                   iteration)

            ####################
            logger.info(
                'Step4: pitting against previous generation in parallel...')
            # transfer weights of previous generation and current generation to the remote actors, and ask them to pit.
            for signal_queue in self.remote_actors_signal_queues:
                signal_queue.put({"task": "pitting"})
            previous_wins, current_wins, draws = 0, 0, 0
            for _ in range(self.args.actors_num):
                (pwins_, cwins_,
                 draws_) = self.remote_actors_return_queue.get()["pitting"]
                previous_wins += pwins_
                current_wins += cwins_
                draws += draws_

            logger.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' %
                        (current_wins, previous_wins, draws))
            if previous_wins + current_wins == 0 or float(current_wins) / (
                    previous_wins + current_wins) < self.args.updateThreshold:
                logger.info('REJECTING NEW MODEL')
                self.current_agent.restore(
                    os.path.join(self.args.checkpoint, 'temp.pth.tar'))
            else:
                logger.info('ACCEPTING NEW MODEL')
                self.current_agent.save(
                    os.path.join(self.args.checkpoint, 'best.pth.tar'))
            self.current_agent.save(
                os.path.join(self.args.checkpoint,
                             self.getCheckpointFile(iteration)))

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(
            folder,
            self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadModel(self):
        self.current_agent.restore(
            os.path.join(self.args.load_folder_file[0],
                         self.args.load_folder_file[1]))

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0],
                                 self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            logger.warning(
                "File {} with trainExamples not found!".format(examplesFile))
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            logger.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            logger.info('Loading done!')
