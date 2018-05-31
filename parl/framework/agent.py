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
from parl.common.error_handling import check_type_error


class DataProcessorBase(object):
    __metaclass__ = ABCMeta

    def __init__(self, specs):
        self.specs = specs
        pass

    @abstractmethod
    def process_prediction_inputs(self, inputs, states):
        pass

    @abstractmethod
    def process_learning_inputs(self, data):
        pass


class AgentHelperBase(object):
    """
    AgentHelper abstracts some part of Agent's data processing and the I/O 
    communication between Agent and ComputationWrapper. It receives a
    Communicator from one ComputationWrapper and uses it to send data to the
    ComputationWrapper.
    """
    __metaclass__ = ABCMeta

    def __init__(self, name, communicator):
        self.name = name
        self.comm = communicator

    @abstractmethod
    def predict(self, inputs, states=dict()):
        """
        Process the inputs (if necessary), send data to ComputationWrapper for
        prediction, and receive the outcome.

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
        Send data to ComputationWrapper for learning and receive learning
        return (if any). 

        Depends on users' need, this function can be called in three ways:
        1. In Agent's run_one_episode
        2. In store_data(), e.g., learning once every few steps
        3. As a separate thread, e.g., using experience replay
        """
        pass


class AgentBase(Process):
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
        super(AgentBase, self).__init__()
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
        assert isinstance(helper, AgentHelperBase)
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

    def run(self):
        """
        Entry function of Agent process.
        """
        for i in range(self.num_games):
            episode_reward = self._run_one_episode()
            if i % 50 == 0:
                print("%d episode reward: %f" % (self.id, episode_reward))
