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
import paddle.fluid as fluid
from parl import layers


class CombinedActionsModel(parl.Model):
    def __init__(self):
        self.fc1 = layers.fc(size=512, act='relu', name='fc1')
        self.fc2 = layers.fc(size=512, act='relu', name='fc2')
        self.fc3 = layers.fc(size=512, act='relu', name='fc3')
        self.fc4 = layers.fc(size=512, act='relu', name='fc4')
        self.fc5 = layers.fc(size=512, act='relu', name='fc5')
        self.fc6 = layers.fc(size=232, act='relu', name='fc6')

    def predict(self, obs):
        out1 = self.fc1(obs)
        out2 = self.fc2(out1)
        out3 = self.fc3(out2)
        out4 = self.fc4(out3)
        out5 = self.fc5(out4)
        out6 = self.fc6(out5)
        return out6


class UnitaryActionModel(parl.Model):
    def __init__(self):
        self.fc1 = layers.fc(size=512, act='relu', name='fc1')
        self.fc2 = layers.fc(size=512, act='relu', name='fc2')
        self.fc3 = layers.fc(size=512, act='relu', name='fc3')
        self.fc4 = layers.fc(size=512, act='relu', name='fc4')
        self.fc5 = layers.fc(size=512, act='relu', name='fc5')
        self.fc6 = layers.fc(size=500, act='relu', name='fc6')

        self.month_embedding = layers.embedding(
            size=[12, 64], name='emb_month')
        self.hour_embedding = layers.embedding(size=[24, 64], name='emb_hour')

    def predict(self, obs):
        dense_input = obs[:, :-2]
        month = obs[:, -2].astype('long')
        hour = obs[:, -1].astype('long')
        fluid.layers.reshape(month, (-1, 1), inplace=True)
        fluid.layers.reshape(hour, (-1, 1), inplace=True)

        month_emb = self.month_embedding(month)
        hour_emb = self.hour_embedding(hour)
        output = fluid.layers.concat(
            input=[dense_input, month_emb, hour_emb], axis=1)

        output1 = self.fc1(output)
        output2 = self.fc2(output1)
        output3 = self.fc3(output2)
        output4 = self.fc4(output3)
        output5 = self.fc5(output4)
        output6 = self.fc6(output5)
        return output6
