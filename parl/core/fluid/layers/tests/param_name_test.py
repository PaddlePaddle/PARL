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

import parl
from paddle import fluid
from parl import layers
import unittest


class MyNetWork(parl.Model):
    def __init__(self):
        self.fc1 = layers.fc(100)
        self.fc2 = layers.fc(100)
        self.fc3 = layers.fc(100, bias_attr=False)
        self.fc4 = layers.fc(100, param_attr=False)
        self.fc5 = layers.fc(100, name="fc", bias_attr=False)
        self.fc6 = layers.fc(100, param_attr=fluid.initializer.Xavier())
        self.embedding = layers.embedding((100, 128))
        self.embedding_custom = layers.embedding((100, 128),
                                                 name="embedding_custom")
        ## although here self.conv2d shares param with self.embedding,
        ## it might be invalid because the param sizes do not match
        self.conv2d = layers.conv2d(
            num_filters=64,
            filter_size=3,
            param_attr=self.embedding.attr_holder.param_attr,
            name="my_conv2d")

        self.batch_norm = layers.batch_norm()


class TestParamName(unittest.TestCase):
    def test_name_number(self):
        net = MyNetWork()

        ## fc1 and fc2 have different parameters
        self.assertEqual(net.fc1.param_name, "fc.w_0")
        self.assertEqual(net.fc2.param_name, "fc.w_1")

        ## fc3 has no bias and fc4 has no param; so the names are None
        self.assertEqual(net.fc3.bias_name, None)
        self.assertEqual(net.fc4.param_name, None)
        self.assertEqual(net.fc4.bias_name, "fc.b_3")

        ## fc5 has a custom name without a bias
        self.assertEqual(net.fc5.param_name, "fc.w_4")
        self.assertEqual(net.fc5.bias_name, None)

        self.assertEqual(net.fc6.param_name, "fc.w_5")

        ## embedding layer has no bias
        self.assertEqual(net.embedding.param_name, "embedding.w_0")
        self.assertEqual(net.embedding.bias_name, None)

        ## embedding layer with a custom name
        self.assertEqual(net.embedding_custom.param_name,
                         "embedding_custom.w_0")

        ## conv2d shares param with embedding; has a custom bias name
        self.assertEqual(net.conv2d.param_name, "embedding.w_0")
        self.assertEqual(net.conv2d.bias_name, "my_conv2d.b_0")

        self.assertSetEqual(
            set(net.batch_norm.all_params_names),
            set([
                'batch_norm.w_0', 'batch_norm.b_0',
                'batch_norm_moving_mean.w_0', 'batch_norm_moving_variance.w_0'
            ]))


if __name__ == '__main__':
    unittest.main()
