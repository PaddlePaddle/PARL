# Third party code
#
# The following code are copied or modified from:
# https://github.com/Glennnnn/Connect4-Python

import math
import random
import sys

import pygame

from utils import *
from MCTS import MCTS
from Coach import Coach
from parl.utils import logger
from connect4_game import Connect4Game

BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

ROW_COUNT = 6
COLUMN_COUNT = 7

PLAYER = 1
AI = -1
EMPTY = 0

WINDOW_LENGTH = 4
SQUARESIZE = 100
RADIUS = int(SQUARESIZE / 2 - 5)


def draw_board(board, screen, height):
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            pygame.draw.rect(screen, BLUE,
                             (c * SQUARESIZE, r * SQUARESIZE + SQUARESIZE,
                              SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, BLACK,
                               (int(c * SQUARESIZE + SQUARESIZE / 2),
                                int(r * SQUARESIZE + SQUARESIZE * 3 / 2)),
                               RADIUS)

    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            if board[r][c] == PLAYER:
                pygame.draw.circle(screen, RED,
                                   (int(c * SQUARESIZE + SQUARESIZE / 2),
                                    int(r * SQUARESIZE + SQUARESIZE * 3 / 2)),
                                   RADIUS)
            elif board[r][c] == AI:
                pygame.draw.circle(screen, YELLOW,
                                   (int(c * SQUARESIZE + SQUARESIZE / 2),
                                    int(r * SQUARESIZE + SQUARESIZE * 3 / 2)),
                                   RADIUS)
    pygame.display.update()


game = Connect4Game()
board = game._base_board
game_over = False
current_board = board.np_pieces
game.display(current_board)

pygame.init()

width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT + 1) * SQUARESIZE
size = (width, height)
screen = pygame.display.set_mode(size)
draw_board(current_board, screen, height)
my_font = pygame.font.SysFont('monospace', 75)
turn = random.choice([PLAYER, AI])

args = dotdict({
    'load_folder_file': ('./saved_model', 'best_model'),
    'numMCTSSims': 800,
    'cpuct': 4,
})
logger.info('Loading checkpoint {}...'.format(args.load_folder_file))
c = Coach(game, args)
c.loadModel()
agent = c.current_agent
mcts = MCTS(game, agent, args)

while not game_over:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

        if event.type == pygame.MOUSEMOTION:
            pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
            posx = event.pos[0]
            if turn == PLAYER:
                pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE / 2)),
                                   RADIUS)

        pygame.display.update()

        if event.type == pygame.MOUSEBUTTONDOWN:
            pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))

            if turn == PLAYER:
                posx = event.pos[0]
                col = int(math.floor(posx / SQUARESIZE))

                if board.is_valid_move(col):
                    current_board, _ = game.getNextState(
                        current_board, turn, col)

                    if game.getGameEnded(current_board, PLAYER) == 1:
                        label = my_font.render('You win !!!', 1, RED)
                        screen.blit(label, (40, 10))
                        game_over = True
                    elif game.getGameEnded(current_board, PLAYER) == 1e-4:
                        label = my_font.render('Draw !!!', 1, RED)
                        screen.blit(label, (40, 10))
                        game_over = True
                    else:
                        turn = -turn

                    game.display(current_board)
                    draw_board(current_board, screen, height)

        if turn == AI and not game_over:
            x = game.getCanonicalForm(current_board, turn)
            col = int(np.argmax(mcts.getActionProb(x, temp=0)))
            # col = np.argmax(pi)
            if board.is_valid_move(col):
                current_board, _ = game.getNextState(current_board, turn, col)
                if game.getGameEnded(current_board, PLAYER) == -1:
                    label = my_font.render('You lose !!!', 1, RED)
                    screen.blit(label, (40, 10))
                    game_over = True
                elif game.getGameEnded(current_board, PLAYER) == 1e-4:
                    label = my_font.render('Draw !!!', 1, RED)
                    screen.blit(label, (40, 10))
                    game_over = True
                else:
                    turn = -turn

                game.display(current_board)
                draw_board(current_board, screen, height)

    if game_over:
        pygame.time.wait(5000)
