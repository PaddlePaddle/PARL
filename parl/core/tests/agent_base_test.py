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
from parl.core.model_base import ModelBase
from parl.core.algorithm_base import AlgorithmBase
from parl.core.agent_base import AgentBase


class MockModel(ModelBase):
    def __init__(self, weights):
        super(MockModel, self).__init__()
        self.weights = weights

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights


class TestAlgorithm(AlgorithmBase):
    def __init__(self):
        self.model1 = MockModel(1)
        self.model2 = MockModel(2)

    def get_weights(self):
        return {
            'model1': [self.model1.get_weights()],
            'model2': {
                'k2': self.model2.get_weights()
            }
        }

    def set_weights(self, params):
        self.model1.set_weights(params['model1'][0])
        self.model2.set_weights(params['model2']['k2'])


class AgentBaseTest(unittest.TestCase):
    def setUp(self):
        alg1 = TestAlgorithm()
        alg2 = TestAlgorithm()
        self.agent1 = AgentBase(alg1)
        self.agent2 = AgentBase(alg2)

    def test_get_weights(self):
        weights = self.agent1.get_weights()
        expected_dict = {'model1': [1], 'model2': {'k2': 2}}
        self.assertDictEqual(weights, expected_dict)

    def test_set_weights(self):
        expected_dict = {'model1': [-1], 'model2': {'k2': -2}}
        self.agent1.set_weights(expected_dict)
        self.assertDictEqual(self.agent1.get_weights(), expected_dict)

    def test_get_and_set_weights_between_agents(self):
        expected_dict = {'model1': [-1], 'model2': {'k2': -2}}
        self.agent1.set_weights(expected_dict)
        new_weights = self.agent1.get_weights()

        self.agent2.set_weights(new_weights)
        self.assertDictEqual(self.agent2.get_weights(), expected_dict)


if __name__ == '__main__':
    unittest.main()
