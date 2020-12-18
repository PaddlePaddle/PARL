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

import unittest
import parl
from parl.core.model_base import ModelBase
from parl.core.algorithm_base import AlgorithmBase


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
        self.model_list1 = (-1, MockModel(3))
        self.model_list2 = [MockModel(4), MockModel(5)]
        self.model_dict1 = {'k1': MockModel(6), 'k2': -2}
        self.model_dict2 = {'k1': MockModel(7), 'k2': MockModel(8)}


class AlgorithmBaseTest(unittest.TestCase):
    def setUp(self):
        self.alg1 = TestAlgorithm()
        self.alg2 = TestAlgorithm()

    def test_get_weights(self):
        weights = self.alg1.get_weights()
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
        self.alg1.set_weights(expected_dict)
        self.assertDictEqual(self.alg1.get_weights(), expected_dict)

    def test_set_weights_with_inconsistent_weights_case1(self):
        inconsistent_weights = {
            'model0': -1,
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
        with self.assertRaises(AssertionError):
            self.alg1.set_weights(inconsistent_weights)

    def test_set_weights_with_inconsistent_weights_case2(self):
        inconsistent_weights = {
            'model1': -1,
            'model2': -2,
            'model_list1': [-3],
            'model_list2': [-4],
            'model_dict1': {
                'k1': -6
            },
            'model_dict2': {
                'k1': -7,
                'k2': -8
            }
        }
        with self.assertRaises(AssertionError):
            self.alg1.set_weights(inconsistent_weights)

    def test_set_weights_with_inconsistent_weights_case3(self):
        inconsistent_weights = {
            'model1': -1,
            'model2': -2,
            'model_list1': [-3],
            'model_list2': [-4, -5],
            'model_dict1': {
                'k1': -6
            },
            'model_dict2': {
                'k1': -7,
            }
        }
        with self.assertRaises(AssertionError):
            self.alg1.set_weights(inconsistent_weights)

    def test_set_weights_with_redundant_weights(self):
        redundant_weights = {
            'model0': 0,
            'model1': -1,
            'model2': -2,
            'model_list1': [-3],
            'model_list2': [-4, -5],
            'model_list3': [44, 55],
            'model_dict1': {
                'k1': -6
            },
            'model_dict2': {
                'k1': -7,
                'k2': -8
            },
            'model_dict3': {
                'k1': -77,
                'k2': -88
            }
        }
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
        self.alg1.set_weights(expected_dict)
        self.assertDictEqual(self.alg1.get_weights(), expected_dict)

    def test_get_and_set_weights_between_algorithms(self):
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
        self.alg1.set_weights(expected_dict)
        new_weights = self.alg1.get_weights()

        self.alg2.set_weights(new_weights)
        self.assertDictEqual(self.alg2.get_weights(), expected_dict)


if __name__ == '__main__':
    unittest.main()
