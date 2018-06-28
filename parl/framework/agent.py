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

from abc import ABCMeta, abstractmethod
from multiprocessing import Process, Value
import numpy as np
from parl.common.communicator import AgentCommunicator
from parl.common.data_process import DataProcessor
from parl.common.replay_buffer import NoReplacementQueue, ReplayBuffer


class AgentHelper(object):
    """
    AgentHelper abstracts some part of Agent's data processing and the I/O 
    communication between Agent and ComputationWrapper. It receives a
    Communicator from one ComputationWrapper and uses it to send data to the
    ComputationWrapper.
    """
    __metaclass__ = ABCMeta

    def __init__(self, name, communicator):
        assert isinstance(communicator, AgentCommunicator)
        self.name = name
        self.comm = communicator

    @abstractmethod
    def predict(self, inputs, states):
        """
        Process the input data (if necessary), send them to `ComputationWrapper`
        for prediction, and receive the outcome.

        Args:
            inputs(dict): data used for prediction. It is caller's job 
            to make sure inputs contains all data needed and they are in the 
            right form.
        """
        pass

    def store_data(self, data):
        """
        Store the past experience for later use, e.g., experience replay.

        Args:
            data(dict): data to store. 
        """
        pass

    @abstractmethod
    def learn(self):
        """
        Sample data from past experiences and send them to `ComputationWrapper`
        for learning. Optionally, it receives learning outcomes sent back from
        CW and does some processing.

        Depends on users' need, this function can be called in three ways:
        1. In Agent's run_one_episode
        2. In store_data(), e.g., learning once every few steps
        3. As a separate thread, e.g., using experience replay
        """
        pass


class OnPolicyHelper(AgentHelper):
    """
    On-policy helper. It calls `learn()` every `sample_interval`
    steps.

    While waiting for learning return, the calling `Agent` is blocked.
    """

    def __init__(self, name, communicator, specs, sample_interval=5):
        super(OnPolicyHelper, self).__init__(name, communicator)
        self.sample_interval = sample_interval
        # NoReplacementQueue used to store past experience.
        # TODO: support sequence sampling 
        self.exp_queue = NoReplacementQueue(sample_seq=False)
        self.counter = 0
        self.data_proc = DataProcessor(specs)

    def predict(self, inputs, states=dict()):
        data = self.data_proc.process_prediction_data(inputs, states)
        self.comm.put_prediction_data((data, 1))
        ret = self.comm.get_prediction_return()
        self.counter += 1
        return ret

    def store_data(self, data):
        isinstance(data, dict)
        episode_end = data["episode_end"]
        self.exp_queue.add(**data)
        if episode_end or self.counter % self.sample_interval == 0:
            self.learn()

    def learn(self):
        self.counter = 0
        exp_seqs, size = self.exp_queue.sample()
        data = self.data_proc.process_learning_data(exp_seqs)
        self.comm.put_training_data((data, size))
        self.comm.get_training_return()


class ExpReplayHelper(AgentHelper):
    """
    Example of applying experience replay. It starts a separate threads to
    run learn().
    """

    def __init__(self, name, communicator, specs, capacity, batch_size):
        super(ExpReplayHelper, self).__init__(name, communicator)
        # replay buffer for experience replay
        self.replay_buffer = ReplayBuffer(capacity)
        self.batch_size = batch_size
        self.data_proc = DataProcessor()
        # the thread that will run learn()
        self.learning_thread = Thread(target=self.learn)
        # prevent race on the replay_buffer
        self.lock = Lock()
        # flag to signal learning_thread to stop
        self.exit_flag = False
        self.learning_thread.start()

    def predict(self, inputs, states=dict()):
        data = self.data_proc.process_prediction_data(inputs, states)
        self.comm.put_prediction_data((data, 1))
        ret = self.comm.get_prediction_return()
        return ret

    def store_data(self, data):
        isinstance(data, dict)
        with self.lock:
            self.replay_buffer.add(self.exp_cls(**data))

    def learn(self):
        """
        This function should be invoked in a separate thread. Once called, it
        keeps sampling data from the replay buffer until exit_flag is signaled.
        """
        # keep running until exit_flag is signaled
        while not self.exit_flag.value:
            exp_seqs = []
            total_size = 0
            with self.lock:
                for s in self.replay_buffer(self.batch_size):
                    exps, size = replay_buffer.get_experiences(s)
                    exp_seqs.append(exps)
                    total_size += size
            data = self.data_proc.process_learning_data(exp_seqs)
            self.comm.put_training_data((data, total_size))
            ret = self.comm.get_training_return()


class Agent(Process):
    """
    Agent implements the control flow and logics of how Robot interacts with 
    the environment and does computation. It is a subclass of Process. The entry
    function of the Agent process is run().

    Agent has the following members:
    env: the environment
    num_games:  number of games to run
    helpers:    a dictionary of AgentHelper, each corresponds to one 
                ComputationTask
    log_q:      communication channel between Agent and the centralized logger
    exit_flag:  signal when the Agent should stop. It is usually set by some 
                other process.
    """
    __metaclass__ = ABCMeta

    def __init__(self, env, num_games):
        super(Agent, self).__init__()
        self.id = -1  # just created, not added to the Robot yet
        self.env = env
        self.num_games = num_games
        self.helpers = {}
        self.log_q = []
        self.exit_flag = Value('i', 0)
        self.daemon = True

    def add_helper(self, helper):
        """
        Add an AgentHelper, with the its name (also the name of its
        correspoding ComputationTask) as key.
        """
        assert isinstance(helper, AgentHelper)
        self.helpers[helper.name] = helper

    def set_log_queue(log_q):
        self.log_q = log_q

    @abstractmethod
    def _run_one_episode(self):
        """
        This function implements the control flow of running one episode, which
        includes:
        1. The interaction with the environment
        2. Calls AgentHelper's interfaces to process the data 
        """
        pass

    def predict(self, alg_name, inputs, states=dict()):
        return self.helpers[alg_name].predict(inputs, states)

    def store_data(self, alg_name, data):
        self.helpers[alg_name].store_data(data)

    def learn(self, alg_name):
        self.helpers[alg_name].learn()

    def run(self):
        """
        Entry function of Agent process.
        """
        # TODO: use logger to handle statistics
        episode_reward = []
        for i in range(self.num_games):
            episode_reward.append(self._run_one_episode())
            if i % 50 == 0:
                print("%d episode reward: %f" %
                      (self.id, sum(episode_reward) / len(episode_reward)))
            if len(episode_reward) == 25:
                episode_reward.pop(0)
