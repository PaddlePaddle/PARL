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

import numpy as np
import parl
import os
from alphazero_agent import create_agent
from MCTS import MCTS
from Arena import Arena
from utils import win_loss_draw


@parl.remote_class(wait=False)
class Actor(object):
    def __init__(self, game, args, seed):
        np.random.seed(seed)
        os.environ['OMP_NUM_THREADS'] = "1"
        self.game = game
        self.args = args

        # neural network of previous generation
        self.previous_agent = create_agent(self.game, cuda=False)
        # neural network of current generation
        self.current_agent = create_agent(self.game, cuda=False)

        # MCTS of previous generation
        self.previous_mcts = MCTS(
            self.game, self.previous_agent, self.args, dirichlet_noise=True)
        # MCTS of current generation
        self.current_mcts = MCTS(
            self.game, self.current_agent, self.args, dirichlet_noise=True)

    def self_play(self, current_weights, game_num):
        """Collecting training data by self-play.
        
        Args:
            current_weights (numpy.array): latest weights of neural network
            game_num (int): game number of self-play

        Returns:
            train_examples (list): examples of the form (canonicalBoard, currPlayer, pi,v)
        """

        # update weights of current neural network with latest weights
        self.current_agent.set_weights(current_weights)

        train_examples = []
        for _ in range(game_num):
            # reset node state of MCTS
            self.current_mcts = MCTS(
                self.game, self.current_agent, self.args, dirichlet_noise=True)
            train_examples.extend(self._executeEpisode())
        return train_examples

    def pitting(self, previous_weights, current_weights, games_num):
        """Fighting between previous generation agent and current generation agent

        Args:
            previous_weights (numpy.array): weights of previous generation neural network
            current_weights (numpy.array): weights of current generation neural network
            game_num (int): game number of fighting 

        Returns:
            tuple of (game number of previous agent won, game number of current agent won, game number of draw)
        """
        # update weights of previous and current neural network
        self.previous_agent.set_weights(previous_weights)
        self.current_agent.set_weights(current_weights)

        # reset node state of MCTS
        self.previous_mcts = MCTS(self.game, self.previous_agent, self.args)
        self.current_mcts = MCTS(self.game, self.current_agent, self.args)

        arena = Arena(
            lambda x: np.argmax(self.previous_mcts.getActionProb(x, temp=0)),
            lambda x: np.argmax(self.current_mcts.getActionProb(x, temp=0)),
            self.game)
        previous_wins, current_wins, draws = arena.playGames(games_num)

        return (previous_wins, current_wins, draws)

    def evaluate_test_dataset(self, current_weights, test_dataset):
        """Evaluate performance of latest neural nerwork
        
        Args:
            current_weights (numpy.array): latest weights of neural network
            test_dataset (list): game number of self-play

        Returns:
            tuple of (number of perfect moves, number of good moves)
        """
        # update weights of current neural network with latest weights
        self.current_agent.set_weights(current_weights)

        perfect_move_count, good_move_count = 0, 0
        for data in test_dataset:
            self.current_mcts = MCTS(self.game, self.current_agent, self.args)

            x = self.game.getCanonicalForm(data['board'], data['player'])
            agent_move = int(
                np.argmax(self.current_mcts.getActionProb(x, temp=0)))

            moves = data["move_score"]
            perfect_score = max(moves)
            perfect_moves = [i for i in range(7) if moves[i] == perfect_score]

            if agent_move in perfect_moves:
                perfect_move_count += 1
            if win_loss_draw(
                    moves[agent_move]) == win_loss_draw(perfect_score):
                good_move_count += 1

        return (perfect_move_count, good_move_count)

    def _executeEpisode(self):
        """

        This function executes one episode of self-play, starting with player 1.
        As the game goes on, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThresholdStep, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThresholdStep)

            pi = self.current_mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:  # board, pi
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(
                board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)

            if r != 0:
                return [(x[0], x[2], r * ((-1)**(x[1] != self.curPlayer)))
                        for x in trainExamples]
