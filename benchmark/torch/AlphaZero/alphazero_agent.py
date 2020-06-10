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

import os
import numpy as np
import parl
import torch
import torch.optim as optim

from tqdm import tqdm
from utils import *
from connect4_model import Connect4Model

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 5,
    'batch_size': 64,
    'num_channels': 64,
})


class AlphaZero(parl.Algorithm):
    def __init__(self, model):
        self.model = model

    def learn(self, boards, target_pis, target_vs, optimizer):
        self.model.train()  # train mode

        # compute model output
        out_log_pi, out_v = self.model(boards)

        pi_loss = -torch.sum(target_pis * out_log_pi) / target_pis.size()[0]

        v_loss = torch.sum(
            (target_vs - out_v.view(-1))**2) / target_vs.size()[0]

        total_loss = pi_loss + v_loss

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return total_loss, pi_loss, v_loss

    def predict(self, board):
        self.model.eval()  # eval mode

        with torch.no_grad():
            log_pi, v = self.model(board)

        pi = torch.exp(log_pi)
        return pi, v


def create_agent(game, cuda=True):
    cuda = cuda and torch.cuda.is_available()

    model = Connect4Model(game, args)
    if cuda:
        model.cuda()

    algorithm = AlphaZero(model)

    alphazero_agent = AlphaZeroAgent(algorithm, game, cuda)
    return alphazero_agent


class AlphaZeroAgent(parl.Agent):
    def __init__(self, algorithm, game, cuda):
        super(AlphaZeroAgent, self).__init__(algorithm)
        self.cuda = cuda
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

    def learn(self, examples):
        """
        Args:
            examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.algorithm.model.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))

            batch_count = int(len(examples) / args.batch_size)

            pbar = tqdm(range(batch_count), desc='Training Net')
            for _ in pbar:
                sample_ids = np.random.randint(
                    len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                if self.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(
                    ), target_pis.contiguous().cuda(), target_vs.contiguous(
                    ).cuda()

                total_loss, pi_loss, v_loss = self.algorithm.learn(
                    boards, target_pis, target_vs, optimizer)

                # record loss with tqdm
                pbar.set_postfix(Loss_pi=pi_loss.item(), Loss_v=v_loss.item())

    def predict(self, board):
        """
        Args:
            board (np.array): input board

        Return:
            pi (np.array): probability of actions
            v (np.array): estimated value of input
        """
        # preparing input
        board = torch.FloatTensor(board.astype(np.float64))
        if self.cuda:
            board = board.contiguous().cuda()
        board = board.view(1, self.board_x, self.board_y)

        pi, v = self.algorithm.predict(board)

        return pi.data.cpu().numpy()[0], v.data.cpu().numpy()[0]


def create_agent(game, cuda=True):
    cuda = cuda and torch.cuda.is_available()

    model = Connect4Model(game, args)
    if cuda:
        model.cuda()

    algorithm = AlphaZero(model)

    alphazero_agent = AlphaZeroAgent(algorithm, game, cuda)
    return alphazero_agent
