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
from Queue import Empty, Full
from threading import Thread, Lock
from parl.common.communicator import CTCommunicator, AgentCommunicator
from parl.common.utils import concat_dicts, split_dict


class ComputationWrapper(object):
    """
    This class batches the data from the Agent side (i.e., sent by AgentHelper)
    and sends them to ComputationTask. At the time of results returned, it
    splits the batched results and sends them back to AgentHelper.
    """

    def __init__(self, name, ct, min_batchsize, max_batchsize, timeout,
                 sample_method, **kwargs):
        self.name = name
        self.ct = ct
        self.min_batchsize = min_batchsize
        self.max_batchsize = max_batchsize
        self.timeout = timeout
        self.helper_creator = (
            lambda comm: sample_method(name, comm, **kwargs))
        self.comm = CTCommunicator(self.timeout)
        self.comms = {}
        self.prediction_thread = Thread(target=self._prediction_loop)
        self.training_thread = Thread(target=self._training_loop)
        self.lock = Lock()
        self.exit_flag = True

    def _pack_data(self, data):
        """
        Pack a list of data into one dict according to network's inputs specs.

        Args:
            data(list of dict): a list of data collected from Agents.
        """
        assert isinstance(data, list)
        data, sizes = zip(*data)
        starts = [0]
        for size in sizes:
            starts.append(size + starts[-1])
        batch = {}
        for k in data[0].iterkeys():
            batch[k] = concat_dicts((d[k] for d in data))

        return batch, starts

    def _unpack_data(self, batch_dict_list, starts):
        """
        Unpack the dict into a list of dict, by slicing each value in the dict.        

        Args:
            batch_dict(dict): 
        """
        ret = []
        for batch_dict in batch_dict_list:
            dict_list = split_dict(batch_dict, starts)
            assert not ret or len(ret[-1]) == len(dict_list)
            ret.append(dict_list)

        return zip(*ret)

    def create_helper(self, agent_id):
        comm = self.__create_communicator(agent_id)
        return self.helper_creator(comm)

    def __create_communicator(self, agent_id):
        """
        Creates a `AgentCommunicator` with this `ComputationWrapper`'s
        data communication channels (training and prediction Queues). Once an
        `AgentHelper` of some `Agent` accepts this communicator (i.e., the 
        `AgentHelper` is created with it), the `Agent` can exchange data with 
        this CW through the communicator.
        """
        self.comms[agent_id] = AgentCommunicator(
            agent_id, self.comm.training_q, self.comm.prediction_q,
            self.timeout)
        return self.comms[agent_id]

    def do_one_prediction(self, data):
        data, starts = self._pack_data(data)
        with self.lock:
            ret = self.ct.predict(**data)
        ret = self._unpack_data(ret, starts)
        return ret

    def do_one_training(self, data):
        data, starts = self._pack_data(data)
        with self.lock:
            ret = self.ct.learn(**data)
        #ret = self._unpack_data(ret, starts)
        return ret

    def _prediction_loop(self):
        while not self.exit_flag:
            agent_ids = []
            data = []
            try:
                while not agent_ids or not self.comm.prediction_q.empty():
                    agent_id, d = self.comm.get_prediction_data()
                    agent_ids.append(agent_id)
                    data.append(d)
            except Empty as e:
                continue
            ret = self.do_one_prediction(data)
            assert len(ret) == len(agent_ids)
            try:
                for i in range(len(agent_ids)):
                    self.comm.prediction_return(ret[i],
                                                self.comms[agent_ids[i]])
            except Full as e:
                continue

    def _training_loop(self):
        while not self.exit_flag:
            agent_ids = []
            data = []
            try:
                while (len(agent_ids) < self.min_batchsize or
                       len(agent_ids) < self.max_batchsize and
                       not self.comm.training_q.empty()):
                    agent_id, d = self.comm.get_training_data()
                    agent_ids.append(agent_id)
                    data.append(d)
            except Empty as e:
                continue
            # TODO: handle ret as a list, one element for each agent
            ret = self.do_one_training(data)
            try:
                for i in range(len(agent_ids)):
                    self.comm.training_return(ret, self.comms[agent_ids[i]])
            except Full as e:
                continue

    def run(self):
        self.exit_flag = False
        self.prediction_thread.start()
        self.training_thread.start()

    def stop(self):
        self.exit_flag = True
        self.prediction_thread.join()
        self.training_thread.join()
