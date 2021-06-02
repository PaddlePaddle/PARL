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
import numpy as np

from utils import *
from tqdm import tqdm
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
        # compute model output
        out_log_pi, out_v = self.model(boards)

        pi_loss = -paddle.sum(target_pis * out_log_pi) / target_pis.shape[0]

        out_v = paddle.reshape(out_v, [-1])
        v_loss = paddle.sum((target_vs - out_v)**2) / target_vs.shape[0]

        total_loss = pi_loss + v_loss

        # compute gradient and do SGD step
        optimizer.clear_grad()
        total_loss.backward()
        optimizer.step()

        return total_loss, pi_loss, v_loss

    def predict(self, board):
        with paddle.no_grad():
            log_pi, v = self.model(board)

        pi = paddle.exp(log_pi)
        return pi, v


class AlphaZeroAgent(parl.Agent):
    def __init__(self, algorithm, game):
        super(AlphaZeroAgent, self).__init__(algorithm)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

    def learn(self, examples):
        """
        Args:
            examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = paddle.optimizer.Adam(
            learning_rate=args.lr, parameters=self.alg.model.parameters())

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))

            batch_count = int(len(examples) / args.batch_size)

            pbar = tqdm(range(batch_count), desc='Training Net')
            for _ in pbar:
                sample_ids = np.random.randint(
                    len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = paddle.to_tensor(np.array(boards), dtype='float32')
                target_pis = paddle.to_tensor(np.array(pis), dtype='float32')
                target_vs = paddle.to_tensor(np.array(vs), dtype='float32')

                total_loss, pi_loss, v_loss = self.alg.learn(
                    boards, target_pis, target_vs, optimizer)

                # record loss with tqdm
                pbar.set_postfix(
                    Loss_pi=pi_loss.numpy()[0], Loss_v=v_loss.numpy()[0])

    def predict(self, board):
        """
        Args:
            board (np.array): input board

        Return:
            pi (np.array): probability of actions
            v (np.array): estimated value of input
        """
        # preparing input
        board = paddle.to_tensor(board, dtype='float32')
        board = paddle.reshape(board, [1, self.board_x, self.board_y])

        pi, v = self.alg.predict(board)
        return pi.numpy()[0], v.numpy()[0]


def create_agent(game):
    model = Connect4Model(game, args)
    algorithm = AlphaZero(model)
    alphazero_agent = AlphaZeroAgent(algorithm, game)
    return alphazero_agent
