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
from parl.common.data_process import DataProcessor
from parl.common.replay_buffer import NoReplacementQueue, ReplayBuffer
from parl.common.logging import GameLogEntry
from parl.framework.agent import Agent, AgentHelper


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
        self.comm.put_prediction_data(data)
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
        exp_seqs = self.exp_queue.sample()
        data = self.data_proc.process_learning_data(exp_seqs)
        self.comm.put_training_data(data)
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
        self.comm.put_prediction_data(data)
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
            with self.lock:
                for s in self.replay_buffer(self.batch_size):
                    exps = replay_buffer.get_experiences(s)
                    exp_seqs.append(exps)
            data = self.data_proc.process_learning_data(exp_seqs)
            self.comm.put_training_data(data)
            ret = self.comm.get_training_return()


class SimpleRLAgent(Agent):
    """
    This class serves as a template of simple RL algorithms, which has only one
    ComputationTask, "RL", i.e., using and learning an RL policy. 
    
    By using different AgentHelpers, this Agent can be applied to either on-
    policy or off-policy RL algorithms.
    """

    def __init__(self, env, num_games):
        super(SimpleRLAgent, self).__init__(env, num_games)

    def _run_one_episode(self):
        # sensor_inputs, (prev_)states and actions are all dict
        max_steps = self.env._max_episode_steps
        obs = self.env.reset()
        episode_end = False
        r = 0
        log_entry = GameLogEntry(self.id, 'RL')
        for t in range(max_steps - 1):
            #self.env.render()
            sensor = np.array(obs).astype("float32")
            inputs = dict(sensor=sensor)
            actions, _ = self.helpers['RL'].predict(inputs)
            try:
                a = actions["action"][0][0]
                next_obs, r, next_episode_end, _ = self.env.step(a)
            except Exception:
                print actions
                raise Exception

            r /= 100.0
            log_entry.num_steps += 1
            log_entry.total_reward += r
            data = {}
            data.update(inputs)
            data.update({
                "action": np.array([a]).astype("int32"),
                "reward": np.array([r]).astype("float32"),
                "episode_end": np.array([episode_end]).astype("int32")
            })
            self.helpers['RL'].store_data(data)
            obs = next_obs
            episode_end = next_episode_end
            if next_episode_end:
                data = dict(
                    sensor=np.array(next_obs).astype("float32"),
                    action=np.array([0]).astype("int32"),
                    reward=np.array([0]).astype("float32"),
                    episode_end=np.array([episode_end]).astype("int32"))

                self.helpers['RL'].store_data(data)
                break
        return log_entry.total_reward
        #self.log_q.put(log_entry)
