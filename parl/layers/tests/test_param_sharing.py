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
import pprl.layers as layers
import paddle.fluid as fluid
import numpy as np


class TestParamSharing(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestParamSharing, self).__init__(*args, **kwargs)
        self.fc1 = layers.fc(64, bias_attr=False)
        self.fc2 = layers.fc(64, bias_attr=False)
        self.fc3 = layers.fc(64, name="fc")
        self.fc4 = layers.fc(64, name="fc")
        ## we bind the paras of self.embedding to those of self.fc1
        self.embedding = layers.embedding(
            (100, 64), param_attr=self.fc1.param_attr)

    def test_param_sharing(self):
        """
        Test case for parameter sharing between layers of the same type
        """
        main_program = fluid.Program()
        startup_program = fluid.Program()

        with fluid.program_guard(main_program, startup_program):
            x = layers.data(name='x', shape=[100], dtype="float32")
            y1 = self.fc1(input=x)
            y11 = self.fc1(input=x)
            y2 = self.fc2(input=x)
            y3 = self.fc3(input=x)
            y4 = self.fc4(input=x)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)
        batch_size = 10
        input_x = np.random.uniform(0, 1, [batch_size, 100]).astype("float32")
        outputs = exe.run(main_program,
                          feed={"x": input_x},
                          fetch_list=[y1, y11, y2, y3, y4])

        self.assertEqual(
            np.sum(outputs[0].flatten()), np.sum(outputs[1].flatten()))
        self.assertNotEqual(
            np.sum(outputs[1].flatten()), np.sum(outputs[2].flatten()))
        self.assertNotEqual(
            np.sum(outputs[3].flatten()), np.sum(outputs[4].flatten()))

    def test_manual_param_sharing(self):
        """
        Test case for parameter sharing between layers of different types
        """
        batch_size = 10
        dict_size = 100

        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(main_program, startup_program):
            x = layers.data(name='x', shape=[1], dtype="int")
            cx = layers.cast(
                x=layers.one_hot(
                    input=x, depth=dict_size), dtype="float32")
            ## remove bias because embedding layer does not have one
            y1 = self.fc1(input=cx)
            y2 = self.embedding(input=x)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)
        input_x = np.random.randint(
            dict_size, size=(batch_size, 1)).astype("int")
        outputs = exe.run(main_program,
                          feed={'x': input_x},
                          fetch_list=[y1, y2])

        self.assertEqual(
            np.sum(outputs[0].flatten()), np.sum(outputs[1].flatten()))


if __name__ == "__main__":
    unittest.main()
