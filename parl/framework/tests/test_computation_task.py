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
from parl.framework.algorithm import Model
from parl.framework.computation_task import ComputationTask
import parl.framework.policy_distribution as pd
from parl.layers import common_functions as comf
from parl.algorithm_zoo.simple_algorithms import SimpleAC, SimpleQ
from parl.model_zoo.simple_models import SimpleModelDeterministic, SimpleModelAC, SimpleModelQ
from test_algorithm import TestAlgorithm
import numpy as np
from copy import deepcopy
import unittest
import math


class TestModelCNN(Model):
    def __init__(self, width, height, num_actions):
        super(TestModelCNN, self).__init__()
        self.conv = layers.conv2d(
            num_filters=1, filter_size=3, bias_attr=False)
        self.mlp = comf.MLP([
            dict(
                size=32, act="relu", bias_attr=False), dict(
                    size=16, act="relu", bias_attr=False), dict(
                        size=num_actions, act="softmax", bias_attr=False)
        ])
        self.height = height
        self.width = width

    def get_input_specs(self):
        ## image format CHW
        return [("image", dict(shape=[1, self.height, self.width]))]

    def get_action_specs(self):
        return [("action", dict(shape=[1], dtype="int64"))]

    def policy(self, inputs, states):
        conv = self.conv(input=inputs.values()[0])
        dist = pd.CategoricalDistribution(self.mlp(conv))
        return dict(action=dist), states

    def value(self, inputs, states):
        v_value = layers.fill_constant(
            shape=[inputs.values()[0].shape[0], 1], dtype="float32", value=0)
        return dict(v_value=v_value)


class TestComputationTask(unittest.TestCase):
    def test_predict(self):
        """
        Test case for AC-learning and Q-learning predictions
        """
        num_actions = 4

        def test(input, ct, max):
            action_counter = [0] * num_actions
            total = 2000
            for i in range(total):
                actions, states = ct.predict(inputs=input)
                assert not states, "states should be empty"
                ## actions["action"] is a batch of actions
                for a in actions["action"]:
                    action_counter[a[0]] += 1

            if max:
                ### if max, the first action will always be chosen
                for i in range(num_actions):
                    prob = action_counter[i] / float(sum(action_counter))
                    self.assertAlmostEqual(
                        prob, 1.0 if i == 0 else 0.0, places=1)
            else:
                ### the actions should be uniform
                for i in range(num_actions):
                    prob = action_counter[i] / float(sum(action_counter))
                    self.assertAlmostEqual(prob, 1.0 / num_actions, places=1)

        dims = 100

        ac = SimpleAC(model=SimpleModelAC(
            dims=dims,
            num_actions=num_actions,
            mlp_layer_confs=[
                dict(
                    size=32, act="relu", bias_attr=False), dict(
                        size=16, act="relu", bias_attr=False), dict(
                            size=num_actions, act="softmax", bias_attr=False)
            ]))

        ac_cnn = SimpleAC(model=TestModelCNN(
            width=84, height=84, num_actions=num_actions))

        q = SimpleQ(model=SimpleModelQ(
            dims=dims,
            num_actions=num_actions,
            mlp_layer_confs=[
                dict(
                    size=32, act="relu", bias_attr=False), dict(
                        size=16, act="relu", bias_attr=False), dict(
                            size=num_actions, bias_attr=False)
            ]))

        batch_size = 10
        height, width = 84, 84
        sensor = np.zeros([batch_size, dims]).astype("float32")
        image = np.zeros([batch_size, 1, height, width]).astype("float32")

        ct0 = ComputationTask("ac", algorithm=ac)
        ct1 = ComputationTask("q", algorithm=q)
        ct2 = ComputationTask("ac_cnn", algorithm=ac_cnn)

        test(dict(sensor=sensor), ct0, max=False)
        test(dict(sensor=sensor), ct1, max=True)
        test(dict(image=image), ct2, max=False)

    def test_ct_para_sharing(self):
        """
        Test case for two CTs sharing parameters
        """
        alg = TestAlgorithm(model=SimpleModelDeterministic(
            dims=10, mlp_layer_confs=[dict(size=10)]))
        ct0 = ComputationTask("ct0", algorithm=alg)
        ct1 = ComputationTask("ct1", algorithm=alg)

        batch_size = 10
        sensor = np.random.uniform(
            0, 1, [batch_size, alg.model.dims]).astype("float32")

        outputs0, _ = ct0.predict(inputs=dict(sensor=sensor))
        outputs1, _ = ct1.predict(inputs=dict(sensor=sensor))
        self.assertEqual(
            np.sum(outputs0["continuous_action"].flatten()),
            np.sum(outputs1["continuous_action"].flatten()))

    def test_ct_para_sync(self):
        """
        Test case for two CTs copying parameters
        """

        alg = TestAlgorithm(model=SimpleModelDeterministic(
            dims=10, mlp_layer_confs=[dict(size=10)]))

        ct0 = ComputationTask("ct0", algorithm=alg)
        ct1 = ComputationTask("ct1", algorithm=deepcopy(alg))

        batch_size = 10
        sensor = np.random.uniform(
            0, 1, [batch_size, ct0.alg.model.dims]).astype("float32")

        outputs0, _ = ct0.predict(inputs=dict(sensor=sensor))
        outputs1, _ = ct1.predict(inputs=dict(sensor=sensor))
        self.assertNotEqual(
            np.sum(outputs0["continuous_action"].flatten()),
            np.sum(outputs1["continuous_action"].flatten()))

        ct0.alg.model.sync_paras_to(ct1.alg.model, ct1.alg.gpu_id)

        outputs0, _ = ct0.predict(inputs=dict(sensor=sensor))
        outputs1, _ = ct1.predict(inputs=dict(sensor=sensor))
        self.assertEqual(
            np.sum(outputs0["continuous_action"].flatten()),
            np.sum(outputs1["continuous_action"].flatten()))

    def test_ct_learning(self):
        """
        Test training
        """
        num_actions = 2
        dims = 100
        batch_size = 8
        sensor = np.ones(
            [batch_size, dims]).astype("float32") / dims  # normalize
        next_sensor = np.zeros([batch_size, dims]).astype("float32")

        for on_policy in [True, False]:
            if on_policy:
                alg = SimpleAC(
                    model=SimpleModelAC(
                        dims=dims,
                        num_actions=num_actions,
                        mlp_layer_confs=[
                            dict(
                                size=64, act="relu", bias_attr=False), dict(
                                    size=32, act="relu", bias_attr=False),
                            dict(
                                size=num_actions, act="softmax")
                        ]),
                    hyperparas=dict(lr=1e-1))
                ct = ComputationTask("simple_ac", algorithm=alg)
            else:
                alg = SimpleQ(
                    model=SimpleModelQ(
                        dims=dims,
                        num_actions=num_actions,
                        mlp_layer_confs=[
                            dict(
                                size=64, act="relu", bias_attr=False), dict(
                                    size=32, act="relu", bias_attr=False),
                            dict(size=num_actions)
                        ]),
                    update_ref_interval=100,
                    hyperparas=dict(lr=1e-1))
                ct = ComputationTask("simple_q", algorithm=alg)

            for i in range(1000):
                if on_policy:
                    outputs, _ = ct.predict(inputs=dict(sensor=sensor))
                    actions = outputs["action"]
                else:
                    ## randomly assemble a batch
                    actions = np.random.choice(
                        [0, 1], size=(batch_size, 1),
                        p=[0.5, 0.5]).astype("int")
                rewards = (1 - actions).astype("float32")
                cost = ct.learn(
                    inputs=dict(sensor=sensor),
                    next_inputs=dict(next_sensor=next_sensor),
                    next_episode_end=dict(next_episode_end=np.ones(
                        (batch_size, 1)).astype("float32")),
                    actions=dict(action=actions),
                    rewards=dict(reward=rewards))

            print("final cost: %f" % cost["cost"])

            ### the policy should bias towards the first action
            outputs, _ = ct.predict(inputs=dict(sensor=sensor))
            for a in outputs["action"]:
                self.assertEqual(a[0], 0)


if __name__ == "__main__":
    unittest.main()
