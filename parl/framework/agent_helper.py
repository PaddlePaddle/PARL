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
import copy
from threading import Thread, Lock
from parl.common.replay_buffer import Experience, ExperienceQueue


class AgentHelperBase(object):
    __metaclass__ = ABCMeta

    def __init__(self, name, communicator):
        self.name = name
        self.comm = communicator

    @abstractmethod
    def predict(self, inputs):
        pass

    @abstractmethod
    def store_data(self, data):
        pass

    @abstractmethod
    def learn(self):
        pass


class SimpleOnPolicyHelper(AgentHelperBase):
    def __init__(self, name, communicator, options):
        super(SimpleOnPolicyHelper, self).__init__(name, communicator)
        self.sync_steps = options["sync_steps"]
        self.exp_queue = ExperienceQueue(sample_seq=False)
        self.counter = 0

    def predict(self, inputs):
        self.comm.put_prediction_data(inputs)
        ret = self.comm.get_prediction_return()
        self.counter += 1
        return ret

    def store_data(self, data):
        game_status = data["game_status"]
        self.exp_queue.add(Experience(**data))
        if game_status or self.counter % self.sync_steps == 0:
            self.learn()

    def learn(self):
        self.counter = 0
        exp_seqs = self.exp_queue.sample()
        self.comm.put_training_data(exp_seqs)
        self.comm.get_training_return()


class ExpReplayHelper(AgentHelperBase):
    def __init__(self, communicator, options):
        super(ExpReplayHelper, self).__init__(communicator)
        self.replay_buffer = ReplayBuffer(options["capacity"])
        self.batch_size = options["batch_size"]
        self.exit_flag = False
        self.lock = Lock()
        self.learning_thread = Thread(target=self.learn)
        self.learning_thread.start()

    def predict(self, inputs):
        self.comm.put_prediction_data(inputs)
        ret = self.comm.get_prediction_return()
        return ret

    def store_data(self, data):
        with self.lock:
            self.replay_buffer.add(Experience(**data))

    def learn(self):
        exp_seqs = []
        with self.lock:
            for s in self.replay_buffer(self.batch_size):
                exps = replay_buffer.get_experiences(s)
                exp_seqs.append(exps)
        self.comm.get_training_data(exp_seqs)
        return self.comm.get_training_return()
