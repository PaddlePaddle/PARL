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
import unittest
from paddle import fluid
from parl import layers
from parl.core.fluid.agent import Agent
from parl.core.fluid.algorithm import Algorithm
from parl.core.fluid.model import Model
from parl.utils.machine_info import get_gpu_count


class TestModel(Model):
    def __init__(self):
        self.fc1 = layers.fc(size=256)
        self.fc2 = layers.fc(size=128)

    def policy(self, obs):
        out = self.fc1(obs)
        out = self.fc2(out)
        return out


class TestAlgorithm(Algorithm):
    def __init__(self, model):
        self.model = model

    def predict(self, obs):
        return self.model.policy(obs)


class TestAgent(Agent):
    def __init__(self, algorithm, gpu_id=None):
        super(TestAgent, self).__init__(algorithm, gpu_id)

    def build_program(self):
        self.predict_program = fluid.Program()
        with fluid.program_guard(self.predict_program):
            obs = layers.data(name='obs', shape=[10], dtype='float32')
            output = self.algorithm.predict(obs)
        self.predict_output = [output]

    def predict(self, obs):
        output_np = self.fluid_executor.run(
            self.predict_program,
            feed={'obs': obs},
            fetch_list=self.predict_output)[0]
        return output_np


class AgentBaseTest(unittest.TestCase):
    def setUp(self):
        self.model = TestModel()
        self.algorithm = TestAlgorithm(self.model)

    def test_agent(self):
        if get_gpu_count() > 0:
            agent = TestAgent(self.algorithm)
            obs = np.random.random([3, 10]).astype('float32')
            output_np = agent.predict(obs)
            self.assertIsNotNone(output_np)


if __name__ == '__main__':
    unittest.main()
