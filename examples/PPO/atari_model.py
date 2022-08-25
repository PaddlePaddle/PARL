#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn as nn
import paddle.nn.functional as F


class AtariModel(parl.Model):
    """ The Model for Atari env
    Args:
        obs_space (Box): observation space.
        act_space (Discrete): action space.
    """

    def __init__(self, obs_space, act_space):
        super(AtariModel, self).__init__()

        self.conv1 = nn.Conv2D(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2D(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2D(64, 64, 3, stride=1)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 7 * 7, 512)

        self.fc_pi = nn.Linear(512, act_space.n)
        self.fc_v = nn.Linear(512, 1)

    def value(self, obs):
        """ Get value network prediction
        Args:
            obs (np.array): current observation
        """
        obs = obs / 255.0
        out = F.relu(self.conv1(obs))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))

        out = self.flatten(out)
        out = F.relu(self.fc(out))
        value = self.fc_v(out)
        return value

    def policy(self, obs):
        """ Get policy network prediction
        Args:
            obs (np.array): current observation
        """
        obs = obs / 255.0
        out = F.relu(self.conv1(obs))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))

        out = self.flatten(out)
        out = F.relu(self.fc(out))
        logits = self.fc_pi(out)
        return logits
