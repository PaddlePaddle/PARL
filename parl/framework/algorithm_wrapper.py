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
from random import randint
from threading import Thread
from parl.common import Communicator


class ComputationWrapper(object):
    def __init__(self, name, options):
        self.name = name
        self.options = options
        # TODO: create ComputationTask
        self.ct = []
        self.training_q = Queue()
        self.prediction_q = Queue()
        self.comms = {}
        self.prediction_thread = Thread(target=self._prediction_loop)
        self.training_thread = Thread(target=self._training_loop)
        self.exit_flag = True
        # TODO: get a parser of model specs 

    def _pack_data(self, data):
        """
        Pack a list of data into one dict according to network's inputs specs.

        Args:
            data(list of dict): a list of data collected from Agents.
        """
        # TODO: discuss data format
        pass

    def _unpack_data(self, batch_data):
        """
        Unpack the dict into a list of dict, by slicing each value in the dict.        

        Args:
            batch_data(dict): 
        """
        pass

    def create_communicator(self, agent_id):
        comm = Communicator(agent_id, self.training_q, self.prediction_q)
        self.comms[agent_id] = comm
        return comm

    def _prediction_loop(self):
        while not self.exit_flag:
            agent_ids = []
            data = []
            while not agent_ids or not self.prediction_q.empty():
                agent_id, d = self.prediction_q.get()
                agent_ids.append(agent_id)
                data.append(d)
            data = self._pack_data(data)
            ret = self.ct.predict(data)
            ret = self._unpack_data(ret)
            assert len(ret) == len(agent_ids)
            for i in range(len(agent_ids)):
                self.comms[agent_ids[i]].put_prediction_return(ret[i])

    def _training_loop(self):
        while not self.exit_flag:
            agent_ids = []
            data = []
            while (len(agent_ids) < self.options["min_batchsize"] or
                   len(agent_ids) < self.options["max_batchsize"] and
                   not self.training_q.empty()):
                agent_id, d = self.training_q.get()
                agent_ids.append(agent_id)
                data.append(d)
            data = self._pack_data(data)
            ret = self.ct.learn(data)
            ret = self._unpack_data(ret)
            assert len(ret) == len(agent_ids)
            for i in range(len(agent_ids)):
                self.comms[agent_ids[i]].put_training_return(ret[i])

    def run(self):
        self.exit_flag = False
        self.prediction_thread.start()
        self.training_thread.start()

    def stop(self):
        self.exit_flag = True
        self.prediction_thread.join()
        self.training_thread.join()


class ComputationWrapperForTest(ComputationWrapper):
    def __init__(self, name, options):
        super(ComputationWrapperForTest, self).__init__(name, options)

    def _prediction_loop(self):
        while not self.exit_flag:
            agent_ids = []
            ret = []
            while not agent_ids or not self.prediction_q.empty():
                agent_id, d = self.prediction_q.get()
                agent_ids.append(agent_id)
                ret.append({"actions": randint(0, 1), "states": []})
            assert len(ret) == len(agent_ids)
            for i in range(len(agent_ids)):
                self.comms[agent_ids[i]].put_prediction_return(ret[i])

    def _training_loop(self):
        while not self.exit_flag:
            agent_ids = []
            ret = []
            while (len(agent_ids) < self.options["min_batchsize"] or
                   len(agent_ids) < self.options["max_batchsize"] and
                   not self.training_q.empty()):
                agent_id, d = self.training_q.get()
                agent_ids.append(agent_id)
                ret.append({"td": 0})
            assert len(ret) == len(agent_ids)
            for i in range(len(agent_ids)):
                self.comms[agent_ids[i]].put_training_return(ret[i])
