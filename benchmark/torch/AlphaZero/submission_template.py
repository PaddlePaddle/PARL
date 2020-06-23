# Third party code
#
# The following code are copied or modified from:
# https://github.com/suragnair/alpha-zero-general

import os
os.environ['OMP_NUM_THREADS'] = "1"


# ===== utils.py =====
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


# ===== MCTS.py ======
import math
import time
import numpy as np

EPS = 1e-8


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nn_agent, args, dirichlet_noise=False):
        self.game = game
        self.nn_agent = nn_agent
        self.args = args
        self.dirichlet_noise = dirichlet_noise
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def getActionProb(self, canonicalBoard, temp=1, timelimit=4.9):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        dir_noise = self.dirichlet_noise
        start_time = time.time()
        while time.time() - start_time < timelimit:
            self.search(canonicalBoard, dirichlet_noise=dir_noise)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [
            self.Nsa[(s, a)] if (s, a) in self.Nsa else 0
            for a in range(self.game.getActionSize())
        ]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x**(1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, canonicalBoard, dirichlet_noise=False):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = self.game.stringRepresentation(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nn_agent.predict(canonicalBoard)

            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            if dirichlet_noise:
                self.applyDirNoise(s, valids)
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                print("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        if dirichlet_noise:
            self.applyDirNoise(s, valids)
            sum_Ps_s = np.sum(self.Ps[s])
            self.Ps[s] /= sum_Ps_s  # renormalize
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[
                        (s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(
                            self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(
                        self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[
                (s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v

    def applyDirNoise(self, s, valids):
        dir_values = np.random.dirichlet(
            [self.args.dirichletAlpha] * np.count_nonzero(valids))
        dir_idx = 0
        for idx in range(len(self.Ps[s])):
            if self.Ps[s][idx]:
                self.Ps[s][idx] = (0.75 * self.Ps[s][idx]) + (
                    0.25 * dir_values[dir_idx])
                dir_idx += 1


# ===== connect4_game.py ======
import numpy as np
from collections import namedtuple

DEFAULT_HEIGHT = 6
DEFAULT_WIDTH = 7
DEFAULT_WIN_LENGTH = 4

WinState = namedtuple('WinState', 'is_ended winner')


class Board():
    """
    Connect4 Board.
    """

    def __init__(self,
                 height=None,
                 width=None,
                 win_length=None,
                 np_pieces=None):
        "Set up initial board configuration."
        self.height = height or DEFAULT_HEIGHT
        self.width = width or DEFAULT_WIDTH
        self.win_length = win_length or DEFAULT_WIN_LENGTH

        if np_pieces is None:
            self.np_pieces = np.zeros([self.height, self.width], dtype=np.int)
        else:
            self.np_pieces = np_pieces
            assert self.np_pieces.shape == (self.height, self.width)

    def add_stone(self, column, player):
        "Create copy of board containing new stone."
        available_idx, = np.where(self.np_pieces[:, column] == 0)
        if len(available_idx) == 0:
            raise ValueError(
                "Can't play column %s on board %s" % (column, self))

        self.np_pieces[available_idx[-1]][column] = player

    def get_valid_moves(self):
        "Any zero value in top row in a valid move"
        return self.np_pieces[0] == 0

    def get_win_state(self):
        for player in [-1, 1]:
            player_pieces = self.np_pieces == -player
            # Check rows & columns for win
            if (self._is_straight_winner(player_pieces)
                    or self._is_straight_winner(player_pieces.transpose())
                    or self._is_diagonal_winner(player_pieces)):
                return WinState(True, -player)

        # draw has very little value.
        if not self.get_valid_moves().any():
            return WinState(True, None)

        # Game is not ended yet.
        return WinState(False, None)

    def with_np_pieces(self, np_pieces):
        """Create copy of board with specified pieces."""
        if np_pieces is None:
            np_pieces = self.np_pieces
        return Board(self.height, self.width, self.win_length, np_pieces)

    def _is_diagonal_winner(self, player_pieces):
        """Checks if player_pieces contains a diagonal win."""
        win_length = self.win_length
        for i in range(len(player_pieces) - win_length + 1):
            for j in range(len(player_pieces[0]) - win_length + 1):
                if all(player_pieces[i + x][j + x] for x in range(win_length)):
                    return True
            for j in range(win_length - 1, len(player_pieces[0])):
                if all(player_pieces[i + x][j - x] for x in range(win_length)):
                    return True
        return False

    def _is_straight_winner(self, player_pieces):
        """Checks if player_pieces contains a vertical or horizontal win."""
        run_lengths = [
            player_pieces[:, i:i + self.win_length].sum(axis=1)
            for i in range(len(player_pieces) - self.win_length + 2)
        ]
        return max([x.max() for x in run_lengths]) >= self.win_length

    def __str__(self):
        return str(self.np_pieces)


class Connect4Game(object):
    """
    Connect4 Game class implementing the alpha-zero-general Game interface.

    Use 1 for player1 and -1 for player2.
    """

    def __init__(self,
                 height=None,
                 width=None,
                 win_length=None,
                 np_pieces=None):
        self._base_board = Board(height, width, win_length, np_pieces)

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        return self._base_board.np_pieces

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return (self._base_board.height, self._base_board.width)

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return self._base_board.width

    def getNextState(self, board, player, action):
        """Returns a copy of the board with updated move, original board is unmodified.

        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)

        """
        b = self._base_board.with_np_pieces(np_pieces=np.copy(board))
        b.add_stone(action, player)
        return b.np_pieces, -player

    def getValidMoves(self, board, player):
        """Any zero value in top row in a valid move.

        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        return self._base_board.with_np_pieces(
            np_pieces=board).get_valid_moves()

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
               
        """
        b = self._base_board.with_np_pieces(np_pieces=board)
        winstate = b.get_win_state()
        if winstate.is_ended:
            if winstate.winner is None:
                # draw has very little value.
                return 1e-4
            elif winstate.winner == player:
                return +1
            elif winstate.winner == -player:
                return -1
            else:
                raise ValueError('Unexpected winstate found: ', winstate)
        else:
            # 0 used to represent unfinished game.
            return 0

    def getCanonicalForm(self, board, player):
        """ 
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        return board * player

    def getSymmetries(self, board, pi):
        """Board is left/right board symmetric

        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        return [(board, pi),
                (np.array(board[:, ::-1], copy=True),
                 np.array(pi[::-1], copy=True))]

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return board.tostring()

    @staticmethod
    def display(board):
        print(" -----------------------")
        print(' '.join(map(str, range(len(board[0])))))
        print(board)
        print(" -----------------------")


# ===== connect4_model ======
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#class Connect4Model(parl.Model): # Kaggle doesn't support parl package
class Connect4Model(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(Connect4Model, self).__init__()
        self.conv1 = nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(
            args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(
            args.num_channels, args.num_channels, 3, stride=1)
        self.conv4 = nn.Conv2d(
            args.num_channels, args.num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)
        self.bn4 = nn.BatchNorm2d(args.num_channels)

        self.fc1 = nn.Linear(
            args.num_channels * (self.board_x - 4) * (self.board_y - 4), 128)
        self.fc_bn1 = nn.BatchNorm1d(128)

        self.fc2 = nn.Linear(128, 64)
        self.fc_bn2 = nn.BatchNorm1d(64)

        self.fc3 = nn.Linear(64, self.action_size)

        self.fc4 = nn.Linear(64, 1)

    def forward(self, s):
        #                                                            s: batch_size x board_x x board_y
        s = s.view(-1, 1, self.board_x,
                   self.board_y)  # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(
            self.conv1(s)))  # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(
            self.conv2(s)))  # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(
            s)))  # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn4(self.conv4(
            s)))  # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = s.view(
            -1,
            self.args.num_channels * (self.board_x - 4) * (self.board_y - 4))

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

        return F.log_softmax(pi, dim=1), torch.tanh(v)


# ===== simple agent ======
args = dotdict({
    'dropout': 0.3,
    'num_channels': 64,
})


class SimpleAgent():
    def __init__(self, game, cuda=True):
        self.cuda = cuda and torch.cuda.is_available()
        self.model = Connect4Model(game, args)
        if self.cuda:
            self.model.cuda()

        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

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

        self.model.eval()  # eval mode

        with torch.no_grad():
            log_pi, v = self.model(board)

        pi = torch.exp(log_pi)

        return pi.data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def load_checkpoint(self, buffer):
        map_location = None if self.cuda else 'cpu'
        checkpoint = torch.load(buffer, map_location=map_location)
        self.model.load_state_dict(checkpoint)


# ===== predict function ======
import base64
import io

game = Connect4Game()

# AlphaZero players
agent = SimpleAgent(game)
buffer = io.BytesIO(decoded)  # noqa: F821 'decoded' added in gen_submission.py
agent.load_checkpoint(buffer)
mcts_args = dotdict({'numMCTSSims': 800, 'cpuct': 1.0})
mcts = MCTS(game, agent, mcts_args)


def alphazero_agent(obs, config):
    board = np.reshape(obs.board.copy(), game.getBoardSize()).astype(int)
    board[np.where(board == 2)] = -1

    player = 1
    if obs.mark == 2:
        player = -1

    x = game.getCanonicalForm(board, player)

    action = np.argmax(
        mcts.getActionProb(x, temp=0, timelimit=config.timeout - 0.5))
    return int(action)
