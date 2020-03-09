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
import paddle.fluid as fluid
from parl import layers


class LiftModel(parl.Model):
    def __init__(self, act_dim):
        self.act_dim = act_dim
        self.fc_1 = layers.fc(size=512, act='relu')
        self.fc_2 = layers.fc(size=256, act='relu')
        self.fc_3 = layers.fc(size=128, act='tanh')

        self.value_fc = layers.fc(size=1)
        self.policy_fc = layers.fc(size=act_dim)

    def policy(self, obs):
        """
        Args:
            obs(float32 tensor): shape of (B * obs_dim)

        Returns:
            policy_logits(float32 tensor): shape of (B * act_dim)
        """
        h_1 = self.fc_1(obs)
        h_2 = self.fc_2(h_1)
        h_3 = self.fc_3(h_2)
        policy_logits = self.policy_fc(h_3)
        return policy_logits

    def value(self, obs):
        """
        Args:
            obs(float32 tensor): shape of (B * obs_dim)

        Returns:
            values(float32 tensor): shape of (B,)
        """
        h_1 = self.fc_1(obs)
        h_2 = self.fc_2(h_1)
        h_3 = self.fc_3(h_2)
        values = self.value_fc(h_3)
        values = layers.squeeze(values, axes=[1])
        return values

    def policy_and_value(self, obs):
        """
        Args:
            obs(float32 tensor): shape (B * obs_dim)

        Returns:
            policy_logits(float32 tensor): shape of (B * act_dim)
            values(float32 tensor): shape of (B,)
        """
        h_1 = self.fc_1(obs)
        h_2 = self.fc_2(h_1)
        h_3 = self.fc_3(h_2)
        policy_logits = self.policy_fc(h_3)
        values = self.value_fc(h_3)
        values = layers.squeeze(values, axes=[1])

        return policy_logits, values
