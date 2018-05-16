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


class TestParamName(unittest.TestCase):
    def test_name_number(self):
        self.fc1 = layers.fc(100)
        self.fc2 = layers.fc(100)
        self.fc3 = layers.fc(100, bias_attr=False)
        self.fc4 = layers.fc(100, param_attr=False)
        self.fc5 = layers.fc(100, name="fc", bias_attr=False)
        self.embedding = layers.embedding((100, 128))
        self.embedding_custom = layers.embedding(
            (100, 128), name="embedding_custom")
        ## although here self.conv2d shares param with self.embedding,
        ## it might be invalid because the param sizes do not match
        self.conv2d = layers.conv2d(
            num_filters=64,
            filter_size=3,
            param_attr=self.embedding.param_attr,
            name="my_conv2d")
        self.dynamic_grus = []
        for i in range(5):
            self.dynamic_grus.append(layers.dynamic_gru(50))

        ## fc1 and fc2 have different parameters
        self.assertEqual(self.fc1.param_name, "fc.w_0")
        self.assertEqual(self.fc2.param_name, "fc.w_1")

        ## fc3 has no bias and fc4 has no param; so the names are None
        self.assertEqual(self.fc3.bias_name, None)
        self.assertEqual(self.fc4.param_name, None)
        self.assertEqual(self.fc4.bias_name, "fc.b_3")

        ## fc5 has a custom name without a bias
        self.assertEqual(self.fc5.param_name, "fc.w_4")
        self.assertEqual(self.fc5.bias_name, None)

        ## embedding layer has no bias
        self.assertEqual(self.embedding.param_name, "embedding.w_0")
        self.assertEqual(self.embedding.bias_name, None)

        ## embedding layer with a custom name
        self.assertEqual(self.embedding_custom.param_name,
                         "embedding_custom.w_0")

        ## conv2d shares param with embedding; has a custom bias name
        self.assertEqual(self.conv2d.param_name, "embedding.w_0")
        self.assertEqual(self.conv2d.bias_name, "my_conv2d.b_0")

        for i, gru in enumerate(self.dynamic_grus):
            self.assertEqual(gru.param_name, "dynamic_gru.w_%d" % i)


if __name__ == '__main__':
    unittest.main()
