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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class Connect4Model(parl.Model):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(Connect4Model, self).__init__()
        self.conv1 = nn.Conv2D(1, args.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2D(
            args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2D(
            args.num_channels, args.num_channels, 3, stride=1)
        self.conv4 = nn.Conv2D(
            args.num_channels, args.num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm2D(args.num_channels)
        self.bn2 = nn.BatchNorm2D(args.num_channels)
        self.bn3 = nn.BatchNorm2D(args.num_channels)
        self.bn4 = nn.BatchNorm2D(args.num_channels)

        self.fc1 = nn.Linear(
            args.num_channels * (self.board_x - 4) * (self.board_y - 4), 128)
        self.fc_bn1 = nn.BatchNorm1D(128)

        self.fc2 = nn.Linear(128, 64)
        self.fc_bn2 = nn.BatchNorm1D(64)

        self.fc3 = nn.Linear(64, self.action_size)

        self.fc4 = nn.Linear(64, 1)
        self.tanh = nn.Tanh()

    def forward(self, s):
        """
        Args:
            s(paddle.Tensor): batch_size x board_x x board_y
        """
        # batch_size x 1 x board_x x board_y
        s = paddle.reshape(s, shape=[-1, 1, self.board_x, self.board_y])
        # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))
        # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))
        # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn3(self.conv3(s)))
        # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = F.relu(self.bn4(self.conv4(s)))
        s = paddle.reshape(
            s,
            shape=[
                -1, self.args.num_channels * (self.board_x - 4) *
                (self.board_y - 4)
            ])

        s = F.dropout(
            F.relu(self.fc_bn1(self.fc1(s))),
            p=self.args.dropout,
            training=self.training)  # batch_size x 128
        s = F.dropout(
            F.relu(self.fc_bn2(self.fc2(s))),
            p=self.args.dropout,
            training=self.training)  # batch_size x 64

        pi = self.fc3(s)  # batch_size x action_size
        v = self.fc4(s)  # batch_size x 1

        return F.log_softmax(pi, axis=1), self.tanh(v)
