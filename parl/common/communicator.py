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


class Communicator(object):
    """
    A communicator is responsible for data passing between the simulation side
    (i.e., `AgentHelper`) and the computation side (i.e., `ComputationWrapper`).
    
    Communicator's only members are some Queues, which are the channels between 
    the simulation processes and computation processes. 

    Communicator has timeout mechanism. When the get/put operation does not
    return within the specified `timeout`, `Empty`/`Full` exceptions will be
    thrown and users can catch them for further processing.

    A communicator is created by CW and passed to `AgentHelper`.

    There are two types of communicator that will be derived from this class:
    1. `CTCommunicator`: the communicator used by CW
    2. `AgentCommunicator`: the communicator used by 'AgentHelper'
    """

    def __init__(self, timeout):
        """
        Args:
            timeout(float): timeout for Queue's get and put operations.
        """
        self.timeout = timeout


class CTCommunicator(Communicator):
    """
    The communicator used by CW. It provides the necessary interfaces for CW to 
    get data from and return results to the simulation side .
    """

    def __init__(self, timeout):
        """
        Create `CTCommunicator`.

        """
        super(CTCommunicator, self).__init__(timeout)
        self.training_q = Queue()
        self.prediction_q = Queue()

    def get_training_data(self):
        """
        Get data in the training queue, which are put by agents.
        """
        return self.training_q.get(timeout=self.timeout)

    def training_return(self, data, comm):
        """
        Return the training outcome to the agent through the agent's
        communicator.
        """
        assert isinstance(comm, AgentCommunicator)
        comm.training_return_q.put(data, timeout=self.timeout)

    def get_prediction_data(self):
        """
        Get data in the prediction queue, which are put by agents.
        """
        return self.prediction_q.get(timeout=self.timeout)

    def prediction_return(self, data, comm):
        """
        Return the prediction outcome to the agent through the agent's
        communicator.
        """
        assert isinstance(comm, AgentCommunicator)
        comm.prediction_return_q.put(data, timeout=self.timeout)


class AgentCommunicator(Communicator):
    """
    The communicator used by `AgentHelper`. It provides the necessary
    interfaces for AgentHelper to send data to and get results from the 
    computation side .
    """

    def __init__(self, agent_id, training_q, prediction_q, timeout):
        """
        Create `AgentCommunicator`.

        `AgentCommunicator` is bound to an agent. It owns the return queues and
        the references to training and prediction queues from `CTCommunicator`.

        Args:
            agent_id(int): id of the agent.
            training_q(Queue), prediction_q(Queue): references to Queues owned
            by a `ComputationWrapper`.
            timeout(float): timeout for Queue's get and put operations.
        """
        super(AgentCommunicator, self).__init__(timeout)

        self.agent_id = agent_id
        assert not training_q is None
        assert not prediction_q is None
        # A `AgentCommunicator` does not own `training_q` and `prediction_q`,
        # but only the references of them.
        self.training_q = training_q
        self.prediction_q = prediction_q
        # A `AgentCommunicator` owns the returning Queues
        self.training_return_q = Queue(1)
        self.prediction_return_q = Queue(1)

    def put_training_data(self, exp_seqs):
        isinstance(exp_seqs, list)
        self.training_q.put((self.agent_id, exp_seqs), timeout=self.timeout)

    def get_training_return(self):
        return self.training_return_q.get(timeout=self.timeout)

    def put_prediction_data(self, data):
        isinstance(data, dict)
        self.prediction_q.put((self.agent_id, data), timeout=self.timeout)

    def get_prediction_return(self):
        return self.prediction_return_q.get(timeout=self.timeout)
