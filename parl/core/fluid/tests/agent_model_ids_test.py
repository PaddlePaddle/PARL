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
import parl


class MockModel(parl.Model):
    def __init__(self, weights, model_id=None):
        super(MockModel, self).__init__(model_id)
        self.weights = weights

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights


class TestAlgorithm(parl.Algorithm):
    def __init__(self):
        self.model1 = MockModel(1)
        self.model2 = MockModel(2)
        self.model_list1 = (-1, MockModel(3))
        self.model_list2 = [MockModel(4), MockModel(5)]
        self.model_dict1 = {'k1': MockModel(6), 'k2': -2}
        self.model_dict2 = {'k1': MockModel(7), 'k2': MockModel(8)}


class TestAlgorithm2(parl.Algorithm):
    def __init__(self):
        self.model1 = MockModel(1, model_id='id1')
        self.model2 = MockModel(2, model_id='id2')
        self.model_list1 = (-1, MockModel(3, model_id='id3'))
        self.model_list2 = [
            MockModel(4, model_id='id4'),
            MockModel(5, model_id='id5')
        ]
        self.model_dict1 = {'k1': MockModel(6, model_id='id6'), 'k2': -2}
        self.model_dict2 = {
            'k1': MockModel(7, model_id='id7'),
            'k2': MockModel(8, model_id='id8')
        }


class TestAgent(parl.Agent):
    def build_program(self):
        pass


class AgentBaseTest(unittest.TestCase):
    def setUp(self):
        alg1 = TestAlgorithm()
        alg2 = TestAlgorithm()
        self.agent1 = TestAgent(alg1)
        self.agent2 = TestAgent(alg2)

    def test_get_weights(self):
        weights = self.agent1.get_weights()
        expected_dict = {
            'model1': 1,
            'model2': 2,
            'model_list1': [3],
            'model_list2': [4, 5],
            'model_dict1': {
                'k1': 6
            },
            'model_dict2': {
                'k1': 7,
                'k2': 8
            }
        }
        self.assertDictEqual(weights, expected_dict)

    def test_set_weights(self):
        expected_dict = {
            'model1': -1,
            'model2': -2,
            'model_list1': [-3],
            'model_list2': [-4, -5],
            'model_dict1': {
                'k1': -6
            },
            'model_dict2': {
                'k1': -7,
                'k2': -8
            }
        }
        self.agent1.set_weights(expected_dict)
        self.assertDictEqual(self.agent1.get_weights(), expected_dict)

    def test_get_and_set_weights_between_agents(self):
        expected_dict = {
            'model1': -1,
            'model2': -2,
            'model_list1': [-3],
            'model_list2': [-4, -5],
            'model_dict1': {
                'k1': -6
            },
            'model_dict2': {
                'k1': -7,
                'k2': -8
            }
        }
        self.agent1.set_weights(expected_dict)
        new_weights = self.agent1.get_weights()

        self.agent2.set_weights(new_weights)
        self.assertDictEqual(self.agent2.get_weights(), expected_dict)

    def test_get_model_ids(self):
        alg = TestAlgorithm2()
        agent = TestAgent(alg)
        expected_model_ids = set(['id{}'.format(i + 1) for i in range(8)])
        self.assertSetEqual(expected_model_ids, agent.get_model_ids())

    def test_get_weights_with_model_ids(self):
        weights = self.agent1.get_weights(model_ids=[
            self.agent1.alg.model1.model_id, self.agent1.alg.model_list2[0].
            model_id, self.agent1.alg.model_dict2['k1'].model_id
        ])
        expected_dict = {
            'model1': 1,
            'model_list2': [4],
            'model_dict2': {
                'k1': 7,
            }
        }
        self.assertDictEqual(weights, expected_dict)

    def test_set_weights_with_model_ids(self):
        new_weights = {
            'model1': -1,
            'model_list2': [-4],
            'model_dict2': {
                'k1': -7,
            }
        }
        expected_dict = {
            'model1': -1,
            'model2': 2,
            'model_list1': [3],
            'model_list2': [-4, 5],
            'model_dict1': {
                'k1': 6
            },
            'model_dict2': {
                'k1': -7,
                'k2': 8
            }
        }

        self.agent1.set_weights(
            new_weights,
            model_ids=[
                self.agent1.alg.model1.model_id,
                self.agent1.alg.model_list2[0].model_id,
                self.agent1.alg.model_dict2['k1'].model_id
            ])
        self.assertDictEqual(self.agent1.get_weights(), expected_dict)

    def test_get_and_set_weights_between_agents_with_model_ids(self):
        agent1_model_ids = [
            self.agent1.alg.model1.model_id,
            self.agent1.alg.model_list2[0].model_id,
            self.agent1.alg.model_dict2['k1'].model_id
        ]
        agent2_model_ids = [
            self.agent2.alg.model1.model_id,
            self.agent2.alg.model_list2[0].model_id,
            self.agent2.alg.model_dict2['k1'].model_id
        ]
        new_weights = {
            'model1': -1,
            'model_list2': [-4],
            'model_dict2': {
                'k1': -7,
            }
        }
        expected_dict = {
            'model1': -1,
            'model2': 2,
            'model_list1': [3],
            'model_list2': [-4, 5],
            'model_dict1': {
                'k1': 6
            },
            'model_dict2': {
                'k1': -7,
                'k2': 8
            }
        }
        self.agent1.set_weights(new_weights, agent1_model_ids)
        agent1_weights = self.agent1.get_weights(agent1_model_ids)

        self.agent2.set_weights(agent1_weights, agent2_model_ids)
        self.assertDictEqual(self.agent2.get_weights(), expected_dict)


if __name__ == '__main__':
    unittest.main()
