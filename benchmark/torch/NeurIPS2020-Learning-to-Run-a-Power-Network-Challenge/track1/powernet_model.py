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

import torch
import torch.nn as nn


class CombinedActionModel(nn.Module):
    def __init__(self):
        super(CombinedActionModel, self).__init__()
        self.linear_1 = nn.Sequential(nn.Linear(636, 512), nn.ReLU())
        self.linear_2 = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
        self.linear_3 = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
        self.linear_4 = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
        self.linear_5 = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
        self.linear_6 = nn.Linear(512, 232)

    def forward(self, input):
        output = self.linear_1(input)
        output = self.linear_2(output)
        output = self.linear_3(output)
        output = self.linear_4(output)
        output = self.linear_5(output)
        return self.linear_6(output)


class UnitaryActionModel(nn.Module):
    def __init__(self):
        super(UnitaryActionModel, self).__init__()
        self.linear_1 = nn.Sequential(
            nn.Linear(198 - 59 + 64 * 2, 512), nn.ReLU())  # remove p
        self.linear_2 = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
        self.linear_3 = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
        self.linear_4 = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
        self.linear_5 = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
        self.linear_6 = nn.Linear(512, 500)

        self.month_embedding = nn.Embedding(12, 64)
        self.hour_embedding = nn.Embedding(24, 64)

    def forward(self, input):
        dense_input = input[:, :-2]
        month = input[:, -2].long()
        hour = input[:, -1].long()

        month_emb = self.month_embedding(month)
        hour_emb = self.hour_embedding(hour)

        output = torch.cat((dense_input, month_emb, hour_emb), dim=1)

        output = self.linear_1(output)
        output = self.linear_2(output)
        output = self.linear_3(output)
        output = self.linear_4(output)
        output = self.linear_5(output)
        return self.linear_6(output)
