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
from parl.framework.algorithm import Model, RLAlgorithm
from parl.layers import common_functions as comf
from parl.model_zoo.simple_models import SimpleModelDeterministic
import numpy as np
from copy import deepcopy
import unittest


class TestAlgorithm(RLAlgorithm):
    def __init__(self, model):
        super(TestAlgorithm, self).__init__(
            model, hyperparas=dict(), gpu_id=-1)


class TestAlgorithmParas(unittest.TestCase):
    def test_sync_paras_in_one_program(self):
        """
        Test case for copying parameters
        """

        alg1 = TestAlgorithm(model=SimpleModelDeterministic(
            dims=10, mlp_layer_confs=[dict(size=10)]))
        alg2 = deepcopy(alg1)

        batch_size = 10
        sensor = np.random.uniform(
            0, 1, [batch_size, alg1.model.dims]).astype("float32")

        program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(program, startup_program):
            x = layers.data(name='x', shape=[alg1.model.dims], dtype="float32")
            try:
                # too eary to sync before the layers are created
                alg1.model.sync_paras_to(alg2.model, alg2.gpu_id)
                self.assertTrue(False)  # you shouldn't be here
            except:
                pass
            ## first let the program generates the actual variables by using the
            ## layer functions (before this step the layers haven't been instantiated yet!)
            ## the call of predict() function already covers all the layers
            y0, _ = alg1.predict(inputs=dict(sensor=x), states=dict())
            y1, _ = alg2.predict(inputs=dict(sensor=x), states=dict())

        ######################
        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(startup_program)

        outputs = exe.run(
            program,
            feed={'x': sensor},
            ## y and y1 are two dictionaries
            fetch_list=y0.values() + y1.values())

        self.assertNotEqual(
            np.sum(outputs[0].flatten()), np.sum(outputs[1].flatten()))

        ## do the copying
        alg1.model.sync_paras_to(alg2.model, alg2.gpu_id)

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
        alg1 = TestAlgorithm(model=SimpleModelDeterministic(
            dims=10, mlp_layer_confs=[dict(size=10)]))
        alg2 = deepcopy(alg1)

        batch_size = 10
        sensor = np.random.uniform(
            0, 1, [batch_size, alg1.model.dims]).astype("float32")

        startup_program = fluid.Program()
        program1 = fluid.Program()
        program2 = fluid.Program()

        with fluid.program_guard(program1, startup_program):
            x1 = layers.data(
                name='x', shape=[alg1.model.dims], dtype="float32")
            y1, _ = alg1.predict(inputs=dict(sensor=x1), states=dict())

        with fluid.program_guard(program2, startup_program):
            x2 = layers.data(
                name='x', shape=[alg1.model.dims], dtype="float32")
            y2, _ = alg2.predict(inputs=dict(sensor=x2), states=dict())

        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(startup_program)

        alg1.model.sync_paras_to(alg2.model, alg2.gpu_id)

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
