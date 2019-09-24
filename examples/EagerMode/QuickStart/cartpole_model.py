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


class CartpoleModel(fluid.dygraph.Layer):
    def __init__(self, name_scope, act_dim):
        super(CartpoleModel, self).__init__(name_scope)
        hid1_size = act_dim * 10
        self.fc1 = fluid.FC('fc1', hid1_size, act='tanh')
        self.fc2 = fluid.FC('fc2', act_dim, act='softmax')

    def forward(self, obs):
        out = self.fc1(obs)
        out = self.fc2(out)
        return out
