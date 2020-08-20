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
import paddle.fluid as fluid
import parl
import unittest
from parl import layers


class MyNetWork(parl.Model):
    def __init__(self):
        self.fc1 = layers.fc(64, bias_attr=False)
        self.fc2 = layers.fc(64, bias_attr=False)
        self.fc3 = layers.fc(64, name="fc")
        self.fc4 = layers.fc(64, name="fc")
        self.embedding = layers.embedding(
            (100, 64), param_attr=self.fc1.attr_holder.param_attr)
        self.created_param = layers.create_parameter(
            shape=[100],
            dtype='float32',
            default_initializer=fluid.initializer.Uniform(low=-1.0, high=1.0))
        self.batch_norm = layers.batch_norm()


class TestParamSharing(unittest.TestCase):
    def test_param_sharing(self):
        """
        Test case for parameter sharing between layers of the same type
        """
        net = MyNetWork()
        ## we bind the paras of embedding to those of fc1
        batch_size = 10
        dict_size = 100
        input_cx = np.random.uniform(0, 1, [batch_size, 100]).astype("float32")
        input_x = np.random.randint(
            dict_size, size=(batch_size, 1)).astype("int64")
        #################################

        main_program1 = fluid.Program()
        with fluid.program_guard(main_program1):
            x = layers.data(name='x', shape=[100], dtype="float32")
            y1 = net.fc1(input=x)
            y11 = net.fc1(input=x)
            y2 = net.fc2(input=x)
            y3 = net.fc3(input=x)
            y4 = net.fc4(input=x)

        main_program2 = fluid.Program()
        with fluid.program_guard(main_program2):
            x_ = layers.data(name='x', shape=[1], dtype="int64")
            cx_ = layers.cast(
                x=layers.one_hot(input=x_, depth=dict_size), dtype="float32")
            y1_ = net.fc1(input=cx_)
            y2_ = net.embedding(input=x_)

            x1_ = layers.data(name='x1', shape=[100], dtype="float32")
            y3_ = net.fc1(input=x1_)

        #### we run the startup program only once to make sure
        #### only one para init across the two programs
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        ######################################################

        outputs = exe.run(
            main_program1,
            feed={"x": input_cx},
            fetch_list=[y1, y11, y2, y3, y4])
        old_y1 = outputs[0]
        self.assertEqual(
            np.sum(outputs[0].flatten()), np.sum(outputs[1].flatten()))
        self.assertNotEqual(
            np.sum(outputs[1].flatten()), np.sum(outputs[2].flatten()))
        self.assertNotEqual(
            np.sum(outputs[3].flatten()), np.sum(outputs[4].flatten()))

        outputs = exe.run(
            main_program2,
            feed={
                'x': input_x,
                'x1': input_cx
            },
            fetch_list=[y1_, y2_, y3_])

        ### test two different layers sharing the same para matrix
        self.assertEqual(
            np.sum(outputs[0].flatten()), np.sum(outputs[1].flatten()))
        ### test if the same layer can have the same parameters across two different programs
        self.assertEqual(
            np.sum(outputs[2].flatten()), np.sum(old_y1.flatten()))

    def test_param_sharing_with_create_parameter(self):
        """
        Test case for parameter sharing of create_parameter op
        """
        net = MyNetWork()

        main_program1 = fluid.Program()
        with fluid.program_guard(main_program1):
            x = layers.data(name='x', shape=[100], dtype="float32")
            out1 = x + net.created_param()

        main_program2 = fluid.Program()
        with fluid.program_guard(main_program2):
            x = layers.data(name='x', shape=[100], dtype="float32")
            out2 = x + net.created_param()

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())

        input_np = np.random.uniform(0, 1, [1, 100]).astype("float32")
        out1_np = exe.run(
            main_program1, feed={"x": input_np}, fetch_list=[out1])[0]
        out2_np = exe.run(
            main_program2, feed={"x": input_np}, fetch_list=[out2])[0]
        self.assertEqual(np.sum(out1_np.flatten()), np.sum(out2_np.flatten()))

    def test_param_sharing_with_batch_norm(self):
        """
        Test case for batch_norm layer
        """
        net = MyNetWork()

        main_program1 = fluid.Program()
        with fluid.program_guard(main_program1):
            x = layers.data(name='x', shape=[32, 128, 128], dtype="float32")
            hid1 = net.fc1(x)
            out1 = net.batch_norm(hid1)

        main_program2 = fluid.Program()
        with fluid.program_guard(main_program2):
            x = layers.data(name='x', shape=[32, 128, 128], dtype="float32")
            hid1 = net.fc1(x)
            out2 = net.batch_norm(hid1)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())

        input_np = np.random.uniform(0, 1, [1, 32, 128, 128]).astype("float32")
        out1_np = exe.run(
            main_program1, feed={"x": input_np}, fetch_list=[out1])[0]
        out2_np = exe.run(
            main_program2, feed={"x": input_np}, fetch_list=[out2])[0]
        self.assertEqual(np.sum(out1_np.flatten()), np.sum(out2_np.flatten()))


if __name__ == "__main__":
    unittest.main()
