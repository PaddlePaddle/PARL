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

line_num = 186
observe_dim = 639
label_dim = 1000


class PowerNetModel(nn.Module):
    def __init__(self):
        super(PowerNetModel, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(1219, 1024), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(1024, 900), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(900, 800), nn.ReLU())
        self.fc4 = nn.Sequential(nn.Linear(800, 700), nn.ReLU())
        self.fc5 = nn.Sequential(nn.Linear(700, 512), nn.ReLU())
        self.fc6 = nn.Sequential(nn.Linear(512, 1000), nn.ReLU())

        self.month_embedding = nn.Embedding(13, 256)
        self.hour_embedding = nn.Embedding(24, 256)
        self.line_status_embedding = nn.Embedding(400, 256)

    def forward(self, obs):
        continuous_obs = obs[:, 0:-2 - line_num]
        line_status = torch.as_tensor(
            obs[:, -2 - line_num:-2], dtype=torch.long)
        line_status = line_status.reshape(-1, 1)
        month = torch.as_tensor(obs[:, -1], dtype=torch.long)
        hour = torch.as_tensor(obs[:, -2], dtype=torch.long)

        month_emb = self.month_embedding(month)
        hour_emb = self.hour_embedding(hour)
        line_status_emb = self.line_status_embedding(line_status)
        line_status_emb = line_status_emb.reshape(-1, 186, 256)
        line_status_emb_sum = torch.sum(line_status_emb, dim=1)

        output = torch.cat(
            [continuous_obs, hour_emb, month_emb, line_status_emb_sum], dim=1)

        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        output = self.fc4(output)
        output = self.fc5(output)
        output = self.fc6(output)
        return output
