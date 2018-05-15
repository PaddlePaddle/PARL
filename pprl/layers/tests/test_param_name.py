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
        self.conv2d = layers.conv2d(
            num_filters=64,
            filter_size=3,
            name="my_conv2d",
            set_paras=self.embedding.parameters())
        self.dynamic_grus = []
        for i in range(5):
            self.dynamic_grus.append(layers.dynamic_gru(50))

        ## fc1 and fc2 have different parameters
        self.assertEqual(self.fc1.param_name, "fc_0.w")
        self.assertEqual(self.fc2.param_name, "fc_1.w")

        ## fc3 has no bias and fc4 has no param; so the names are None
        self.assertEqual(self.fc3.bias_name, None)
        self.assertEqual(self.fc4.param_name, None)

        ## fc5 has a custom name without a bias
        ## fc5 has a different param name with fc1
        self.assertEqual(self.fc5.param_name, "fc_0_.w")
        self.assertEqual(self.fc5.bias_name, None)

        ## embedding layer has no bias
        self.assertEqual(self.embedding.param_name, "embedding_0.w")
        self.assertEqual(self.embedding.bias_name, None)

        ## embedding layer with a custom name; the custom id is 1 up to this point
        self.assertEqual(self.embedding_custom.param_name,
                         "embedding_custom_1_.w")

        ## conv2d shares param with embedding; has a custom bias name; the custom id is 2 now
        self.assertEqual(self.conv2d.param_name, "embedding_0.w")
        self.assertEqual(self.conv2d.bias_name, "my_conv2d_2_.wbias")

        for i, gru in enumerate(self.dynamic_grus):
            self.assertEqual(gru.param_name, "dynamic_gru_%d.w" % i)


if __name__ == '__main__':
    unittest.main()
