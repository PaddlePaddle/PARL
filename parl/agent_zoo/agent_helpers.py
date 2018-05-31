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

import numpy as np
from threading import Thread, Lock
from parl.common.error_handling import check_type_error
from parl.common.replay_buffer import NoReplacementQueue, ReplayBuffer
from parl.framework.agent import AgentHelperBase, DataProcessorBase


class RLDataProcessor(DataProcessorBase):
    def __init__(self, specs):
        super(RLDataProcessor, self).__init__(specs)

    def process_prediction_inputs(self, inputs, states):
        assert bool(inputs)
        new_inputs = {}
        for k in inputs:
            new_inputs[k] = np.stack([inputs[k]])
        data = dict(inputs=new_inputs)
        if bool(states):
            new_states = {}
            for k in inputs:
                new_states[k] = np.stack([new_states[k]])
            data["states"] = new_states
        return data

    def process_learning_inputs(self, exp_seqs):
        def append_data(data, attr, exp_seq):
            if attr.startswith("next_"):
                l = [getattr(exp.next_exp, attr[5:]) for exp in exp_seq]
            else:
                l = [getattr(exp, attr) for exp in exp_seq]
            if type(l[0]) is list:
                data[attr] = data[attr] + l if attr in data else l
            else:
                data[attr] = np.concatenate(
                    [data[attr], np.stack(l)]) if attr in data else np.stack(l)

        data = {}
        for k in self.specs.iterkeys():
            data[k] = {}
            for attr in self.specs[k]:
                for exp_seq in exp_seqs:
                    append_data(data[k], attr, exp_seq)
        return data


class OnPolicyHelper(AgentHelperBase):
    """
    Example of on-policy helper. It calls learn() every sample_interval steps.
    While waiting for learning return, the Agent is blocked.
    """

    def __init__(self, name, communicator, specs, sample_interval=5):
        super(OnPolicyHelper, self).__init__(name, communicator)
        self.sample_interval = sample_interval
        # NoReplacementQueue used to store past experience.
        self.exp_queue = NoReplacementQueue(sample_seq=False)
        self.counter = 0
        self.data_proc = RLDataProcessor(specs)

    def predict(self, inputs, states=dict()):
        check_type_error(dict, type(inputs))
        check_type_error(dict, type(states))
        data = self.data_proc.process_prediction_inputs(inputs, states)
        self.comm.put_prediction_data(data)
        ret = self.comm.get_prediction_return()
        self.counter += 1
        return ret

    def store_data(self, data):
        check_type_error(dict, type(data))
        episode_end = data["episode_end"]
        self.exp_queue.add(**data)
        if episode_end or self.counter % self.sample_interval == 0:
            self.learn()

    def learn(self):
        self.counter = 0
        exp_seqs = self.exp_queue.sample()
        data = self.data_proc.process_learning_inputs(exp_seqs)
        self.comm.put_training_data(data)
        self.comm.get_training_return()


class ExpReplayHelper(AgentHelperBase):
    """
    Example of applying experience replay. It starts a separate threads to
    run learn().
    """

    def __init__(self, name, communicator, capacity, batch_size):
        super(ExpReplayHelper, self).__init__(name, data_proc, communicator)
        # replay buffer for experience replay
        self.replay_buffer = ReplayBuffer(capacity)
        self.batch_size = batch_size
        self.data_proc = RLDataProcessor()
        # the thread that will run learn()
        self.learning_thread = Thread(target=self.learn)
        # prevent race on the replay_buffer
        self.lock = Lock()
        # flag to signal learning_thread to stop
        self.exit_flag = False
        self.learning_thread.start()

    def predict(self, inputs, states=dict()):
        check_type_error(dict, type(inputs))
        check_type_error(dict, type(states))
        data = self.data_proc.process_prediction_inputs(inputs, states)
        self.comm.put_prediction_data(data)
        ret = self.comm.get_prediction_return()
        return ret

    def store_data(self, data):
        check_type_error(dict, type(data))
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
            with self.lock:
                for s in self.replay_buffer(self.batch_size):
                    exps = replay_buffer.get_experiences(s)
                    exp_seqs.append(exps)
            data = self.data_proc.process_learning_inputs(exp_seqs)
            self.comm.put_training_data(data)
            ret = self.comm.get_training_return()
