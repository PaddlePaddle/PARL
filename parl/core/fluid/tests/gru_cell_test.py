#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from parl import layers
from copy import deepcopy
import unittest
import paddle.fluid as fluid
import parl
from parl.utils import get_gpu_count
import numpy as np

batch_size = 5
rnn_hidden_dim = 32
n_actions = 9
episode_len = 3
n_agents = 3
obs_shape = 42


class GRUModel(parl.Model):
    ''' GRU-based policy model.
    '''

    def __init__(self):
        self.fc1 = layers.fc(size=rnn_hidden_dim, name='fc1')
        self.gru = layers.GRUCell(hidden_size=rnn_hidden_dim, name='gru')
        self.fc2 = layers.fc(size=n_actions, name='fc2')

    def __call__(self, inputs, hidden_state):
        """Args:
            inputs:       (batch_size * n_agents, rnn_hidden_dim)
            hidden_state: (batch_size, rnn_hidden_dim)
        """
        x = fluid.layers.relu(self.fc1(inputs))
        h, _ = self.gru(x, hidden_state)
        q = self.fc2(h)
        return q, h


class GRUModelTest(unittest.TestCase):
    def setUp(self):
        self.gru_model = GRUModel()
        self.target_model = deepcopy(self.gru_model)
        self.target_model_2 = deepcopy(self.gru_model)

        gpu_count = get_gpu_count()
        if gpu_count > 0:
            place = fluid.CUDAPlace(0)
        else:
            place = fluid.CPUPlace()
        self.executor = fluid.Executor(place)

    def test_model_copy(self):
        self.assertNotEqual(self.gru_model.gru.param_candidate_name,
                            self.target_model.gru.param_candidate_name)
        self.assertNotEqual(self.gru_model.gru.param_gate_name,
                            self.target_model.gru.param_gate_name)
        self.assertNotEqual(self.gru_model.gru.bias_candidate_name,
                            self.target_model.gru.bias_candidate_name)
        self.assertNotEqual(self.gru_model.gru.bias_gate_name,
                            self.target_model.gru.bias_gate_name)

    def test_model_copy_with_multi_copy(self):
        self.assertNotEqual(self.target_model.gru.param_candidate_name,
                            self.target_model_2.gru.param_candidate_name)
        self.assertNotEqual(self.target_model.gru.param_gate_name,
                            self.target_model_2.gru.param_gate_name)
        self.assertNotEqual(self.target_model.gru.bias_candidate_name,
                            self.target_model_2.gru.bias_candidate_name)
        self.assertNotEqual(self.target_model.gru.bias_gate_name,
                            self.target_model_2.gru.bias_gate_name)

    def test_sync_weights_in_one_program(self):
        pred_program = fluid.Program()
        with fluid.program_guard(pred_program):
            last_hidden_states = fluid.data(
                name='last_hidden_states',
                shape=[batch_size, n_agents, rnn_hidden_dim])
            target_last_hidden_states = fluid.data(
                name='target_last_hidden_states',
                shape=[batch_size, n_agents, rnn_hidden_dim])
            obs_batch = fluid.data(
                name='obs_batch',
                shape=[batch_size, episode_len, n_agents, obs_shape])

            hidden_states = last_hidden_states
            target_hidden_states = target_last_hidden_states

            # origin model
            for t in range(episode_len):
                obs = obs_batch[:, t, :, :]
                obs = fluid.layers.reshape(
                    obs, shape=(batch_size * n_agents, obs_shape))
                hidden_states = layers.reshape(
                    hidden_states,
                    shape=(batch_size * n_agents, rnn_hidden_dim))
                local_q, hidden_states = self.gru_model(obs, hidden_states)

            # target model
            for t in range(episode_len):
                obs = obs_batch[:, t, :, :]
                obs = fluid.layers.reshape(
                    obs, shape=(batch_size * n_agents, obs_shape))
                target_hidden_states = layers.reshape(
                    target_hidden_states,
                    shape=(batch_size * n_agents, rnn_hidden_dim))
                target_local_q, target_hidden_states = self.target_model(
                    obs, target_hidden_states)

        self.executor.run(fluid.default_startup_program())

        N = 10
        random_last_hidden_states = np.zeros(
            (N, batch_size, n_agents, rnn_hidden_dim)).astype('float32')
        random_obs_batch = np.random.random(
            size=(N, batch_size, episode_len, n_agents,
                  obs_shape)).astype('float32')
        for i in range(N):
            last_hidden_states_test = random_last_hidden_states[i]
            obs_batch_test = random_obs_batch[i]
            feed = {
                'last_hidden_states': last_hidden_states_test,
                'target_last_hidden_states': last_hidden_states_test,
                'obs_batch': obs_batch_test,
            }
            fetch_local_q, fetch_hidden_states, fetch_target_local_q, \
                    fetch_target_hidden_states = self.executor.run(
                pred_program,
                feed=feed,
                fetch_list=[
                    local_q, hidden_states, target_local_q,
                    target_hidden_states
                ])
            self.assertFalse(
                (fetch_local_q.flatten() == fetch_target_local_q.flatten()
                 ).any())
            self.assertFalse((fetch_hidden_states.flatten() ==
                              fetch_target_hidden_states.flatten()).any())

        self.gru_model.sync_weights_to(self.target_model)

        random_last_hidden_states = np.zeros(
            (N, batch_size, n_agents, rnn_hidden_dim)).astype('float32')
        random_obs_batch = np.random.random(
            size=(N, batch_size, episode_len, n_agents,
                  obs_shape)).astype('float32')
        for i in range(N):
            last_hidden_states_test = random_last_hidden_states[i]
            obs_batch_test = random_obs_batch[i]
            feed = {
                'last_hidden_states': last_hidden_states_test,
                'target_last_hidden_states': last_hidden_states_test,
                'obs_batch': obs_batch_test,
            }
            fetch_local_q, fetch_hidden_states, fetch_target_local_q,\
                    fetch_target_hidden_states = self.executor.run(
                pred_program,
                feed=feed,
                fetch_list=[
                    local_q, hidden_states, target_local_q,
                    target_hidden_states
                ])

            self.assertTrue(
                (fetch_local_q.flatten() == fetch_target_local_q.flatten()
                 ).all())
            self.assertTrue((fetch_hidden_states.flatten() ==
                             fetch_target_hidden_states.flatten()).all())

    def test_sync_weights_among_programs(self):
        pred_program = fluid.Program()
        pred_program_2 = fluid.Program()
        with fluid.program_guard(pred_program):
            last_hidden_states = fluid.data(
                name='last_hidden_states',
                shape=[batch_size, n_agents, rnn_hidden_dim])
            obs_batch = fluid.data(
                name='obs_batch',
                shape=[batch_size, episode_len, n_agents, obs_shape])

            hidden_states = last_hidden_states

            # origin model
            for t in range(episode_len):
                obs = obs_batch[:, t, :, :]
                obs = fluid.layers.reshape(
                    obs, shape=(batch_size * n_agents, obs_shape))
                hidden_states = layers.reshape(
                    hidden_states,
                    shape=(batch_size * n_agents, rnn_hidden_dim))
                local_q, hidden_states = self.gru_model(obs, hidden_states)

        with fluid.program_guard(pred_program_2):
            target_last_hidden_states = fluid.data(
                name='target_last_hidden_states',
                shape=[batch_size, n_agents, rnn_hidden_dim])
            obs_batch = fluid.data(
                name='obs_batch',
                shape=[batch_size, episode_len, n_agents, obs_shape])

            target_hidden_states = target_last_hidden_states
            # target model
            for t in range(episode_len):
                obs = obs_batch[:, t, :, :]
                obs = fluid.layers.reshape(
                    obs, shape=(batch_size * n_agents, obs_shape))
                target_hidden_states = layers.reshape(
                    target_hidden_states,
                    shape=(batch_size * n_agents, rnn_hidden_dim))
                target_local_q, target_hidden_states = self.target_model(
                    obs, target_hidden_states)

        self.executor.run(fluid.default_startup_program())

        N = 10
        random_last_hidden_states = np.zeros(
            (N, batch_size, n_agents, rnn_hidden_dim)).astype('float32')
        random_obs_batch = np.random.random(
            size=(N, batch_size, episode_len, n_agents,
                  obs_shape)).astype('float32')
        for i in range(N):
            last_hidden_states_test = random_last_hidden_states[i]
            obs_batch_test = random_obs_batch[i]
            feed = {
                'last_hidden_states': last_hidden_states_test,
                'obs_batch': obs_batch_test,
            }
            feed_2 = {
                'target_last_hidden_states': last_hidden_states_test,
                'obs_batch': obs_batch_test,
            }
            fetch_local_q, fetch_hidden_states = self.executor.run(
                pred_program, feed=feed, fetch_list=[local_q, hidden_states])
            fetch_target_local_q, fetch_target_hidden_states = self.executor.run(
                pred_program_2,
                feed=feed_2,
                fetch_list=[target_local_q, target_hidden_states])
            self.assertFalse(
                (fetch_local_q.flatten() == fetch_target_local_q.flatten()
                 ).any())
            self.assertFalse((fetch_hidden_states.flatten() ==
                              fetch_target_hidden_states.flatten()).any())

        self.gru_model.sync_weights_to(self.target_model)

        random_last_hidden_states = np.zeros(
            (N, batch_size, n_agents, rnn_hidden_dim)).astype('float32')
        random_obs_batch = np.random.random(
            size=(N, batch_size, episode_len, n_agents,
                  obs_shape)).astype('float32')
        for i in range(N):
            last_hidden_states_test = random_last_hidden_states[i]
            obs_batch_test = random_obs_batch[i]
            feed = {
                'last_hidden_states': last_hidden_states_test,
                'obs_batch': obs_batch_test,
            }
            feed_2 = {
                'target_last_hidden_states': last_hidden_states_test,
                'obs_batch': obs_batch_test,
            }
            fetch_local_q, fetch_hidden_states = self.executor.run(
                pred_program, feed=feed, fetch_list=[local_q, hidden_states])
            fetch_target_local_q, fetch_target_hidden_states = self.executor.run(
                pred_program_2,
                feed=feed_2,
                fetch_list=[target_local_q, target_hidden_states])
            self.assertTrue(
                (fetch_local_q.flatten() == fetch_target_local_q.flatten()
                 ).all())
            self.assertTrue((fetch_hidden_states.flatten() ==
                             fetch_target_hidden_states.flatten()).all())


if __name__ == '__main__':
    unittest.main()
