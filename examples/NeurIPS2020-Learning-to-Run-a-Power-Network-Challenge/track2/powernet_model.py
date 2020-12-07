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

import parl
import numpy as np
import paddle.fluid as fluid
from parl import layers
import os

line_num = 186
observe_dim = 639
label_dim = 1000


class PowerNetModel(parl.Model):
    def __init__(self):

        self.fc1 = layers.fc(size=1024, act='relu', name='fc1')
        self.fc2 = layers.fc(size=900, act='relu', name='fc2')
        self.fc3 = layers.fc(size=800, act='relu', name='fc3')
        self.fc4 = layers.fc(size=700, act='relu', name='fc4')
        self.fc5 = layers.fc(size=512, act='relu', name='fc5')
        self.fc6 = layers.fc(size=label_dim, act='relu', name='fc6')

        self.month_embedding = layers.embedding(
            size=[13, 256], name='emb_month')
        self.hour_embedding = layers.embedding(size=[24, 256], name='emb_hour')
        self.line_status_embedding = layers.embedding(
            size=[400, 256], name='emb_line_status')

    def predict(self, obs):
        continuous_obs = fluid.layers.slice(
            obs, axes=[1], starts=[0], ends=[-2 - line_num])
        line_status = fluid.layers.slice(
            obs, axes=[1], starts=[-2 - line_num], ends=[-2]).astype('long')
        line_status = fluid.layers.reshape(line_status, shape=(-1, 1))
        month = obs[:, -1].astype('long')
        hour = obs[:, -2].astype('long')
        fluid.layers.reshape(month, (-1, 1), inplace=True)
        fluid.layers.reshape(hour, (-1, 1), inplace=True)

        month_emb = self.month_embedding(month)
        hour_emb = self.hour_embedding(hour)
        line_status_emb = self.line_status_embedding(line_status)
        line_status_emb = fluid.layers.reshape(
            line_status_emb, shape=(-1, line_num, 256))
        line_status_emb_sum = fluid.layers.reduce_sum(line_status_emb, dim=1)
        output = fluid.layers.concat(
            input=[continuous_obs, hour_emb, month_emb, line_status_emb_sum],
            axis=1)

        output1 = self.fc1(output)
        output2 = self.fc2(output1)
        output3 = self.fc3(output2)
        output4 = self.fc4(output3)
        output5 = self.fc5(output4)
        output6 = self.fc6(output5)
        return output6
