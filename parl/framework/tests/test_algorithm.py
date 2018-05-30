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
from paddle.fluid.layer_helper import LayerHelper
import parl.layers as layers
from parl.framework.net import Model, Algorithm, create_algorithm_func, Feedforward
import numpy as np
import unittest


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


class TestAlgorithm1(Algorithm):
    def __init__(self, model_func, num_dims):
        super(TestAlgorithm1, self).__init__(model_func, gpu_id=-1)
        self.mlp = Feedforward([layers.fc(num_dims) for _ in range(1)])

    def _predict(self, policy_states):
        return dict(continuous_action=self.mlp(policy_states.values()[0]))

    def _learn(self, policy_states, next_policy_states, actions, rewards):
        return dict(cost=rewards.values()[0] - rewards.values()[0])


class TestAlgorithm(unittest.TestCase):
    def test_sync_paras_in_one_program(self):
        """
        Test case for copying parameters
        """

        algorithm_func = create_algorithm_func(
            model_class=TestModel1,
            model_args=dict(dims=10),
            algorithm_class=TestAlgorithm1,
            algorithm_args=dict(num_dims=20))
        alg = algorithm_func()
        ref_alg = algorithm_func()

        batch_size = 10
        sensor = np.random.uniform(
            0, 1, [batch_size, alg.model.dims]).astype("float32")

        program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(program, startup_program):
            x = layers.data(name='x', shape=[alg.model.dims], dtype="float32")
            try:
                # too eary to sync before the layers are created
                alg.copy_to(ref_alg)
                self.assertTrue(False)  # you shouldn't be here
            except:
                pass
            ## first let the program generates the actual variables by using the
            ## layer functions (before this step the layers haven't been instantiated yet!)
            ## the call of predict() function already covers all the layers
            y0, _ = alg.predict(inputs=dict(sensor=x), states=dict())
            y1, _ = ref_alg.predict(inputs=dict(sensor=x), states=dict())

        ######################
        exe = fluid.Executor(alg.place)
        exe.run(startup_program)

        outputs = exe.run(
            program,
            feed={'x': sensor},
            ## y and y1 are two dictionaries
            fetch_list=y0.values() + y1.values())

        self.assertNotEqual(
            np.sum(outputs[0].flatten()), np.sum(outputs[1].flatten()))

        ## do the copying
        alg.copy_to(ref_alg, ref_alg.place)

        outputs = exe.run(
            program,
            feed={'x': sensor},
            ## y and y1 are two dictionaries
            fetch_list=y0.values() + y1.values())

        self.assertEqual(
            np.sum(outputs[0].flatten()), np.sum(outputs[1].flatten()))

    def test_sync_paras_between_programs(self):
        """
        Test case for copying parameters between two different programs
        """
        algorithm_func = create_algorithm_func(
            model_class=TestModel1,
            model_args=dict(dims=10),
            algorithm_class=TestAlgorithm1,
            algorithm_args=dict(num_dims=20))
        alg = algorithm_func()
        ref_alg = algorithm_func()

        batch_size = 10
        sensor = np.random.uniform(
            0, 1, [batch_size, alg.model.dims]).astype("float32")

        startup_program = fluid.Program()
        program1 = fluid.Program()
        program2 = fluid.Program()

        with fluid.program_guard(program1, startup_program):
            x1 = layers.data(name='x', shape=[alg.model.dims], dtype="float32")
            y1, _ = alg.predict(inputs=dict(sensor=x1), states=dict())

        with fluid.program_guard(program2, startup_program):
            x2 = layers.data(name='x', shape=[alg.model.dims], dtype="float32")
            y2, _ = ref_alg.predict(inputs=dict(sensor=x2), states=dict())

        exe = fluid.Executor(alg.place)
        exe.run(startup_program)

        alg.copy_to(ref_alg, ref_alg.place)

        outputs1 = exe.run(program1,
                           feed={'x': sensor},
                           fetch_list=y1.values())
        outputs2 = exe.run(program2,
                           feed={'x': sensor},
                           fetch_list=y2.values())
        self.assertEqual(
            np.sum(outputs1[0].flatten()), np.sum(outputs2[0].flatten()))


if __name__ == "__main__":
    unittest.main()
