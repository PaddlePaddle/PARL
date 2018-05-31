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

import paddle.fluid as fluid
import parl.layers as layers
from parl.framework.net import Model, Algorithm, create_algorithm_func, Feedforward
from parl.framework.computation_task import ComputationTask
from parl.algorithm_zoo.simple_algorithms import SimpleAC, SimpleQ
import numpy as np
import copy
import unittest
import math


class TestModel1(Model):
    def __init__(self, dims):
        super(TestModel1, self).__init__()
        self.dims = dims
        self.fc = layers.fc(dims)

    def get_input_specs(self):
        return [("sensor", dict(shape=[self.dims]))]

    def get_action_specs(self):
        return [("continuous_action", dict(shape=[self.dims]))]

    def perceive(self, inputs, states):
        hidden = self.fc(input=inputs.values()[0])
        return dict(hidden=hidden), states


class TestModel2(Model):
    def __init__(self, dims):
        super(TestModel2, self).__init__()
        self.dims = dims

    def get_input_specs(self):
        return [("sensor", dict(shape=[self.dims]))]

    def get_action_specs(self):
        return [("action", dict(shape=[1], dtype="int64"))]

    def perceive(self, inputs, states):
        return inputs, states


class TestModelCNN(Model):
    def __init__(self, width, height):
        super(TestModelCNN, self).__init__()
        self.conv = layers.conv2d(
            num_filters=1, filter_size=3, bias_attr=False)
        self.height = height
        self.width = width

    def get_input_specs(self):
        ## image format CHW
        return [("image", dict(shape=[1, self.height, self.width]))]

    def get_action_specs(self):
        return [("action", dict(shape=[1], dtype="int64"))]

    def perceive(self, inputs, states):
        ### TODO: model.perceive has to consider the "next_" prefix
        assert "image" in inputs or "next_image" in inputs
        conv = self.conv(input=inputs.values()[0])
        return dict(conv=conv), states


class TestAlgorithm1(Algorithm):
    def __init__(self, num_dims):
        super(TestAlgorithm1, self).__init__(gpu_id=-1)
        self.mlp = Feedforward([layers.fc(num_dims) for _ in range(1)])

    def _predict(self, policy_states):
        return dict(continuous_action=self.mlp(policy_states.values()[0]))

    def _learn(self, policy_states, next_policy_states, actions, rewards):
        return dict(cost=rewards.values()[0] - rewards.values()[0])


class TestComputationTask(unittest.TestCase):
    def test_predict(self):
        """
        Test case for AC-learning and Q-learning predictions
        """

        def test(input, ct, max):
            action_counter = [0] * ct.alg.num_actions
            total = 1000
            for i in range(total):
                actions, states = ct.predict(inputs=input, states=dict())
                assert not states, "states should be empty"
                ## actions["action"] is a batch of actions
                for a in actions["action"]:
                    action_counter[a[0]] += 1

            if max:
                ### if max, the first action will always be chosen
                for i in range(ct.alg.num_actions):
                    prob = action_counter[i] / float(sum(action_counter))
                    self.assertAlmostEqual(
                        prob, 1.0 if i == 0 else 0.0, places=1)
            else:
                ### the actions should be almost uniform
                for i in range(ct.alg.num_actions):
                    prob = action_counter[i] / float(sum(action_counter))
                    self.assertAlmostEqual(
                        prob, 1.0 / ct.alg.num_actions, places=1)

        num_actions = 4
        dims = 100
        mlp_layer_confs = [
            dict(
                size=32, act="relu", bias_attr=False), dict(
                    size=16, act="relu", bias_attr=False)
        ]

        ac_func = create_algorithm_func(
            model_class=TestModel2,
            model_args=dict(dims=dims),
            algorithm_class=SimpleAC,
            algorithm_args=dict(
                num_actions=num_actions, mlp_layer_confs=mlp_layer_confs))

        ac_cnn_func = create_algorithm_func(
            model_class=TestModelCNN,
            model_args=dict(
                width=84, height=84),
            algorithm_class=SimpleAC,
            algorithm_args=dict(
                num_actions=num_actions, mlp_layer_confs=mlp_layer_confs))

        q_func = create_algorithm_func(
            model_class=TestModel2,
            model_args=dict(dims=dims),
            algorithm_class=SimpleQ,
            algorithm_args=dict(num_actions=num_actions,
                                mlp_layer_confs=mlp_layer_confs \
                                + [dict(size=num_actions, bias_attr=False)]))

        batch_size = 10
        height, width = 84, 84
        sensor = np.zeros([batch_size, dims]).astype("float32")
        image = np.zeros([batch_size, 1, height, width]).astype("float32")

        ct0 = ComputationTask(algorithm=ac_func)
        ct1 = ComputationTask(algorithm=q_func)
        ct2 = ComputationTask(algorithm=ac_cnn_func)

        test(dict(sensor=sensor), ct0, max=False)
        test(dict(sensor=sensor), ct1, max=True)
        test(dict(image=image), ct2, max=False)

    def test_ct_para_sharing(self):
        """
        Test case for two CTs sharing parameters
        """
        algorithm_func = create_algorithm_func(
            model_class=TestModel1,
            model_args=dict(dims=10),
            algorithm_class=TestAlgorithm1,
            algorithm_args=dict(num_dims=20))
        alg = algorithm_func()

        ct0 = ComputationTask(algorithm=alg)
        ct1 = ComputationTask(algorithm=alg)

        batch_size = 10
        sensor = np.random.uniform(
            0, 1, [batch_size, alg.model.dims]).astype("float32")

        outputs0, _ = ct0.predict(inputs=dict(sensor=sensor), states=dict())
        outputs1, _ = ct1.predict(inputs=dict(sensor=sensor), states=dict())
        self.assertEqual(
            np.sum(outputs0["continuous_action"].flatten()),
            np.sum(outputs1["continuous_action"].flatten()))

    def test_ct_para_sync(self):
        """
        Test case for two CTs copying parameters
        """
        algorithm_func = create_algorithm_func(
            model_class=TestModel1,
            model_args=dict(dims=10),
            algorithm_class=TestAlgorithm1,
            algorithm_args=dict(num_dims=20))

        ct0 = ComputationTask(algorithm=algorithm_func)
        ct1 = ComputationTask(algorithm=algorithm_func)

        batch_size = 10
        sensor = np.random.uniform(
            0, 1, [batch_size, ct0.alg.model.dims]).astype("float32")

        outputs0, _ = ct0.predict(inputs=dict(sensor=sensor), states=dict())
        outputs1, _ = ct1.predict(inputs=dict(sensor=sensor), states=dict())
        self.assertNotEqual(
            np.sum(outputs0["continuous_action"].flatten()),
            np.sum(outputs1["continuous_action"].flatten()))

        ct0.alg.copy_to(ct1.alg, ct1.alg.place)

        outputs0, _ = ct0.predict(inputs=dict(sensor=sensor), states=dict())
        outputs1, _ = ct1.predict(inputs=dict(sensor=sensor), states=dict())
        self.assertEqual(
            np.sum(outputs0["continuous_action"].flatten()),
            np.sum(outputs1["continuous_action"].flatten()))

    def test_ct_learning(self):
        """
        Test off-policy training
        """
        num_actions = 2
        dims = 100
        batch_size = 8
        sensor = np.ones(
            [batch_size, dims]).astype("float32") / dims  # normalize
        next_sensor = np.zeros([batch_size, dims]).astype("float32")
        mlp_layer_confs = [
            dict(
                size=64, act="relu", bias_attr=False), dict(
                    size=32, act="relu", bias_attr=False),
            dict(size=num_actions)
        ]

        alg_func = create_algorithm_func(
            model_class=TestModel2,
            model_args=dict(dims=dims),
            algorithm_class=SimpleQ,
            algorithm_args=dict(
                num_actions=num_actions,
                mlp_layer_confs=mlp_layer_confs,
                update_ref_interval=100))

        ct = ComputationTask(algorithm=alg_func)

        for i in range(2000):
            ## randomly assemble a batch
            actions = np.random.choice(
                [0, 1], size=(batch_size, 1), p=[0.5, 0.5]).astype("int")
            rewards = copy.deepcopy(1 - actions).astype("float32")
            cost = ct.learn(
                inputs=dict(sensor=sensor),
                next_inputs=dict(next_sensor=sensor),
                states=dict(),
                next_states=dict(),
                actions=dict(action=actions),
                rewards=dict(reward=rewards))

        print("final cost: %f" % cost["cost"])

        ### the policy should bias towards the first action
        outputs, _ = ct.predict(inputs=dict(sensor=sensor), states=dict())
        for a in outputs["action"]:
            self.assertEqual(a[0], 0)


if __name__ == "__main__":
    unittest.main()
