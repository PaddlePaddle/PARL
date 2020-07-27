#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from parl import layers


class AtariModel(parl.Model):
    def __init__(self, act_dim):
        self.act_dim = act_dim

        self.conv1 = layers.conv2d(
            num_filters=32, filter_size=5, stride=1, padding=2, act='relu')
        self.conv2 = layers.conv2d(
            num_filters=32, filter_size=5, stride=1, padding=2, act='relu')
        self.conv3 = layers.conv2d(
            num_filters=64, filter_size=4, stride=1, padding=1, act='relu')
        self.conv4 = layers.conv2d(
            num_filters=64, filter_size=3, stride=1, padding=1, act='relu')

        self.fc1 = layers.fc(size=act_dim)

    def value(self, obs):
        obs = obs / 255.0
        out = self.conv1(obs)
        out = layers.pool2d(
            input=out, pool_size=2, pool_stride=2, pool_type='max')
        out = self.conv2(out)
        out = layers.pool2d(
            input=out, pool_size=2, pool_stride=2, pool_type='max')
        out = self.conv3(out)
        out = layers.pool2d(
            input=out, pool_size=2, pool_stride=2, pool_type='max')
        out = self.conv4(out)
        out = layers.flatten(out, axis=1)

        Q = self.fc1(out)
        return Q
