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

import unittest
import numpy as np
import paddle.fluid as fluid
import parl
from parl import layers


class DQNModel(parl.Model):
    def __init__(self):
        self.fc1 = layers.fc(size=32, act='relu')
        self.fc2 = layers.fc(size=2)

    def value(self, obs):
        x = self.fc1(obs)
        act = self.fc2(x)
        return act


class DQNAgent(parl.Agent):
    def __init__(self, algorithm):
        super(DQNAgent, self).__init__(algorithm)
        self.alg = algorithm

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):
            obs = layers.data(name='obs', shape=[4], dtype='float32')
            self.value = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):
            obs = layers.data(name='obs', shape=[4], dtype='float32')
            action = layers.data(name='act', shape=[1], dtype='int32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(name='next_obs', shape=[4], dtype='float32')
            lr = layers.data(
                name='lr', shape=[1], dtype='float32', append_batch_size=False)
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            self.cost = self.alg.learn(obs, action, reward, next_obs, terminal,
                                       lr)

    def predict(self, obs):
        obs = np.expand_dims(obs, axis=0)
        pred_Q = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.value])[0]
        pred_Q = np.squeeze(pred_Q, axis=0)
        act = np.argmax(pred_Q)
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        lr = 3e-4

        obs = np.expand_dims(obs, axis=0)
        next_obs = np.expand_dims(next_obs, axis=0)
        act = np.expand_dims(act, -1)
        feed = {
            'obs': obs.astype('float32'),
            'act': act.astype('int32'),
            'reward': reward,
            'next_obs': next_obs.astype('float32'),
            'terminal': terminal,
            'lr': np.float32(lr)
        }
        cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.cost])[0]
        return cost


class A3CModel(parl.Model):
    def __init__(self):
        self.fc = layers.fc(size=32, act='relu')

        self.policy_fc = layers.fc(size=2)
        self.value_fc = layers.fc(size=1)

    def policy(self, obs):
        x = self.fc(obs)
        policy_logits = self.policy_fc(x)

        return policy_logits

    def value(self, obs):
        x = self.fc(obs)
        values = self.value_fc(x)
        values = layers.squeeze(values, axes=[1])

        return values

    def policy_and_value(self, obs):
        x = self.fc(obs)
        policy_logits = self.policy_fc(x)
        values = self.value_fc(x)
        values = layers.squeeze(values, axes=[1])

        return policy_logits, values


class A3CAgent(parl.Agent):
    def __init__(self, algorithm):
        super(A3CAgent, self).__init__(algorithm)
        self.alg = algorithm

    def build_program(self):
        self.predict_program = fluid.Program()
        self.value_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.predict_program):
            obs = layers.data(name='obs', shape=[4], dtype='float32')
            self.predict_actions = self.alg.predict(obs)

        with fluid.program_guard(self.value_program):
            obs = layers.data(name='obs', shape=[4], dtype='float32')
            self.values = self.alg.value(obs)

        with fluid.program_guard(self.learn_program):
            obs = layers.data(name='obs', shape=[4], dtype='float32')
            actions = layers.data(name='actions', shape=[], dtype='int64')
            advantages = layers.data(
                name='advantages', shape=[], dtype='float32')
            target_values = layers.data(
                name='target_values', shape=[], dtype='float32')
            lr = layers.data(
                name='lr', shape=[1], dtype='float32', append_batch_size=False)
            entropy_coeff = layers.data(
                name='entropy_coeff',
                shape=[1],
                dtype='float32',
                append_batch_size=False)

            total_loss, pi_loss, vf_loss, entropy = self.alg.learn(
                obs, actions, advantages, target_values, lr, entropy_coeff)
            self.learn_outputs = [total_loss, pi_loss, vf_loss, entropy]

    def predict(self, obs_np):
        obs_np = obs_np.astype('float32')

        predict_actions = self.fluid_executor.run(
            self.predict_program,
            feed={'obs': obs_np},
            fetch_list=[self.predict_actions])[0]
        return predict_actions

    def value(self, obs_np):
        obs_np = obs_np.astype('float32')

        values = self.fluid_executor.run(
            self.value_program, feed={'obs': obs_np},
            fetch_list=[self.values])[0]
        return values

    def learn(self, obs_np, actions_np, advantages_np, target_values_np):
        obs_np = obs_np.astype('float32')
        actions_np = actions_np.astype('int64')
        advantages_np = advantages_np.astype('float32')
        target_values_np = target_values_np.astype('float32')

        lr = 3e-4
        entropy_coeff = 0.

        total_loss, pi_loss, vf_loss, entropy = self.fluid_executor.run(
            self.learn_program,
            feed={
                'obs': obs_np,
                'actions': actions_np,
                'advantages': advantages_np,
                'target_values': target_values_np,
                'lr': np.array([lr], dtype='float32'),
                'entropy_coeff': np.array([entropy_coeff], dtype='float32')
            },
            fetch_list=self.learn_outputs)
        return total_loss, pi_loss, vf_loss, entropy, lr, entropy_coeff


class IMPALAModel(parl.Model):
    def __init__(self):
        self.fc = layers.fc(size=32, act='relu')

        self.policy_fc = layers.fc(size=2)
        self.value_fc = layers.fc(size=1)

    def policy(self, obs):
        x = self.fc(obs)
        policy_logits = self.policy_fc(x)

        return policy_logits

    def value(self, obs):
        x = self.fc(obs)
        values = self.value_fc(x)
        values = layers.squeeze(values, axes=[1])

        return values


class IMPALAAgent(parl.Agent):
    def __init__(self, algorithm):
        super(IMPALAAgent, self).__init__(algorithm)
        self.alg = algorithm

    def build_program(self):
        self.predict_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.predict_program):
            obs = layers.data(name='obs', shape=[4], dtype='float32')
            self.predict_actions = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):
            obs = layers.data(name='obs', shape=[4], dtype='float32')
            actions = layers.data(name='actions', shape=[], dtype='int64')
            behaviour_logits = layers.data(
                name='behaviour_logits', shape=[2], dtype='float32')
            rewards = layers.data(name='rewards', shape=[], dtype='float32')
            dones = layers.data(name='dones', shape=[], dtype='float32')
            lr = layers.data(
                name='lr', shape=[1], dtype='float32', append_batch_size=False)
            entropy_coeff = layers.data(
                name='entropy_coeff',
                shape=[1],
                dtype='float32',
                append_batch_size=False)

            vtrace_loss, kl = self.alg.learn(obs, actions, behaviour_logits,
                                             rewards, dones, lr, entropy_coeff)
            self.learn_outputs = [
                vtrace_loss.total_loss, vtrace_loss.pi_loss,
                vtrace_loss.vf_loss, vtrace_loss.entropy, kl
            ]

    def predict(self, obs_np):
        obs_np = obs_np.astype('float32')

        predict_actions = self.fluid_executor.run(
            self.predict_program,
            feed={'obs': obs_np},
            fetch_list=[self.predict_actions])[0]
        return predict_actions

    def learn(self, obs, actions, behaviour_logits, rewards, dones, lr,
              entropy_coeff):
        total_loss, pi_loss, vf_loss, entropy, kl = self.fluid_executor.run(
            self.learn_program,
            feed={
                'obs': obs,
                'actions': actions,
                'behaviour_logits': behaviour_logits,
                'rewards': rewards,
                'dones': dones,
                'lr': np.array([lr], dtype='float32'),
                'entropy_coeff': np.array([entropy_coeff], dtype='float32')
            },
            fetch_list=self.learn_outputs)
        return total_loss, pi_loss, vf_loss, entropy, kl


class PARLtest(unittest.TestCase):
    def setUp(self):
        # set up DQN test
        DQN_model = DQNModel()
        DQN_alg = parl.algorithms.DQN(DQN_model, act_dim=2, gamma=0.9)
        self.DQN_agent = DQNAgent(DQN_alg)

        # set up A3C test
        A3C_model = A3CModel()
        A3C_alg = parl.algorithms.A3C(A3C_model, vf_loss_coeff=0.)
        self.A3C_agent = A3CAgent(A3C_alg)

        # set up IMPALA test
        IMPALA_model = IMPALAModel()
        IMPALA_alg = parl.algorithms.IMPALA(
            IMPALA_model,
            sample_batch_steps=4,
            gamma=0.9,
            vf_loss_coeff=0.,
            clip_rho_threshold=1.,
            clip_pg_rho_threshold=1.)
        self.IMPALA_agent = IMPALAAgent(IMPALA_alg)

    def test_DQN_predict(self):
        """Test APIs in PARL DQN predict
        """
        obs = np.array([-0.02394919, 0.03114079, 0.01136446, 0.00324496])

        act = self.DQN_agent.predict(obs)

    def test_DQN_learn(self):
        """Test APIs in PARL DQN learn
        """
        obs = np.array([-0.02394919, 0.03114079, 0.01136446, 0.00324496])
        next_obs = np.array([-0.02332638, -0.16414229, 0.01142936, 0.29949173])
        terminal = np.array([False]).astype('bool')
        reward = np.array([1.0]).astype('float32')
        act = np.array([0]).astype('int32')

        cost = self.DQN_agent.learn(obs, act, reward, next_obs, terminal)

    def test_A3C_predict(self):
        """Test APIs in PARL A3C predict
        """
        obs = np.array([-0.02394919, 0.03114079, 0.01136446, 0.00324496])
        obs = np.expand_dims(obs, axis=0)

        logits = self.A3C_agent.predict(obs)

    def test_A3C_value(self):
        """Test APIs in PARL A3C predict
        """
        obs = np.array([-0.02394919, 0.03114079, 0.01136446, 0.00324496])
        obs = np.expand_dims(obs, axis=0)

        values = self.A3C_agent.value(obs)

    def test_A3C_learn(self):
        """Test APIs in PARL A3C learn
        """
        obs = np.array([[-0.02394919, 0.03114079, 0.01136446, 0.00324496]])
        action = np.array([0])
        advantages = np.array([-0.02332638])
        target_values = np.array([1.])

        self.A3C_agent.learn(obs, action, advantages, target_values)

    def test_IMPALA_predict(self):
        """Test APIs in PARL IMPALA predict
        """
        obs = np.array([[-0.02394919, 0.03114079, 0.01136446, 0.00324496]])

        policy = self.IMPALA_agent.predict(obs)

    def test_IMPALA_learn(self):
        """Test APIs in PARL IMPALA learn
        """
        obs = np.array([[-0.02394919, 0.03114079, 0.01136446, 0.00324496],
                        [-0.02394919, 0.03114079, 0.01136446, 0.00324496],
                        [-0.02394919, 0.03114079, 0.01136446, 0.00324496],
                        [-0.02394919, 0.03114079, 0.01136446,
                         0.00324496]]).astype('float32')
        actions = np.array([1, 1, 1, 1]).astype('int32')
        behaviour_logits = np.array([[-1, 1], [-1, 1], [-1, 1],
                                     [-1, 1]]).astype('float32')
        rewards = np.array([0, 0, 0, 0]).astype('float32')
        dones = np.array([False, False, False, False]).astype('float32')
        lr = 3e-4
        entropy_coeff = 0.

        total_loss, pi_loss, vf_loss, entropy, kl = self.IMPALA_agent.learn(
            obs, actions, behaviour_logits, rewards, dones, lr, entropy_coeff)


if __name__ == '__main__':
    unittest.main()
