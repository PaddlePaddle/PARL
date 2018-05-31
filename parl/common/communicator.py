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

from multiprocessing import Queue
from parl.common.error_handling import check_type_error


class CommunicatorBase(object):
    """
    A Communicator is responsible for data passing between AgentHelper and
    ComputationWrapper (CW). Its only members are some Queues, which are the 
    channels between the Agent processes and computation processes.

    Communicator is created by CW and passed to AgentHelper.

    There are two types of Communicator that will be derived from this class:
    1. CommunicatorCT: the Communicator used by CW
    2. CommunicatorAgent: the Communicator used by AgentHelper
    """


class CommunicatorCT(CommunicatorBase):
    """
    The Communicator used by ComputationWrapper (CW). It provides the necessary
    interfaces for CW to get data from and return results to the Agent side .
    """

    def __init__(self, timeout):
        """
        Create Communicator.

        Args:
            timeout(float): timeout for Queue's get and put operations.
        """
        self.timeout = timeout
        self.training_q = Queue()
        self.prediction_q = Queue()

    def get_training_data(self):
        return self.training_q.get(timeout=self.timeout)

    def training_return(self, data, comm):
        assert isinstance(comm, CommunicatorAgent)
        comm.training_return_q.put(data, timeout=self.timeout)

    def get_prediction_data(self):
        return self.prediction_q.get(timeout=self.timeout)

    def prediction_return(self, data, comm):
        assert isinstance(comm, CommunicatorAgent)
        comm.prediction_return_q.put(data, timeout=self.timeout)


class CommunicatorAgent(CommunicatorBase):
    """
    The Communicator used by AgentHelper. It provides the necessary
    interfaces for AgentHelper to send data to and get results from the 
    Agent side .
    """

    def __init__(self, agent_id, training_q, prediction_q, timeout):
        """
        Create Communicator.

        Args:
            training_q(Queue), prediction_q(Queue): references to Queues owned
            by a ComputationWrapper.
            timeout(float): timeout for Queue's get and put operations.
        """
        self.agent_id = agent_id
        self.timeout = timeout
        assert not training_q is None
        assert not prediction_q is None
        self.training_q = training_q
        self.prediction_q = prediction_q
        # A Communicator owns the returning Queues
        self.training_return_q = Queue(1)
        self.prediction_return_q = Queue(1)

    def put_training_data(self, exp_seqs):
        check_type_error(dict, type(exp_seqs))
        self.training_q.put((self.agent_id, exp_seqs), timeout=self.timeout)

    def get_training_return(self):
        return self.training_return_q.get(timeout=self.timeout)

    def put_prediction_data(self, data):
        check_type_error(dict, type(data))
        self.prediction_q.put((self.agent_id, data), timeout=self.timeout)

    def get_prediction_return(self):
        return self.prediction_return_q.get(timeout=self.timeout)
