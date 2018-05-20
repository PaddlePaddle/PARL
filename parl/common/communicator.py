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


class Communicator(object):
    """
    A Communicator is responsible for data passing between Agent and
    ComputationWrapper. A Communicator is created by ComputationWrapper and
    passed to one Agent.
    """

    #TODO decide whether to use task_done and Queue.join()
    def __init__(self, agent_id, training_q, prediction_q):
        """
        Create Communicator.

        Args:
            agent_id(integer): The id of the Agent this Communicator is
            connected to.
            training_q(Queue), prediction_q(Queue): references to Queues owned
            by some ComputationWrapper.
        """
        self.agent_id = agent_id
        self.training_q = training_q
        self.prediction_q = prediction_q
        # A Communicator owns the returning Queues
        self.training_return_q = Queue(1)
        self.prediction_return_q = Queue(1)

    def put_training_data(self, exp_seqs):
        check_type_error(list, type(exp_seqs))
        self.training_q.put((self.agent_id, exp_seqs))

    def put_training_return(self, data):
        self.training_return_q.put(data)

    def get_training_return(self):
        return self.training_return_q.get()

    def put_prediction_data(self, data):
        check_type_error(dict, type(data))
        self.prediction_q.put((self.agent_id, data))

    def put_prediction_return(self, data):
        self.prediction_return_q.put(data)

    def get_prediction_return(self):
        return self.prediction_return_q.get()
