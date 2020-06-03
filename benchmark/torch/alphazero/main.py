from Coach import Coach
from connect4_game import Connect4Game
from utils import *

from parl.utils import logger


args = dotdict({
    'master_address': 'localhost:8010',     # master address of xparl cluster
    'actors_num': 25,                       # number of remote actors (execute tasks [self-play/pitting/evaluate_test_dataset] in parallel).

    'numIters': 200,                        # total number of iteration 
    'numEps': 500,                          # Number of complete self-play games to simulate during a new iteration.
    'arenaCompare': 50,                     # Number of games to play during arena (pitting) play to determine if new neural network will be accepted.
    'numMCTSSims': 800,                     # Number of games moves for MCTS to simulate.
    'tempThresholdStep': 15,                # temp=1 (Temperature, τ (tau)) if episodeStep < tempThresholdStep, and thereafter uses temp=0.
    'updateThreshold': 0.6,                 # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'cpuct': 4,                             # CPUCT parameter
    'dirichletAlpha': 1.0,                  # alpha parameter of dirichlet noise which is added to the policy (pi)
    'numItersForTrainExamplesHistory': 20,  # history of examples from numItersForTrainExamplesHistory latest iterations (training data)

    'checkpoint': './saved_model/',         # folder to save model and training examples
    'load_model': False,                    # whether to load saved model and training examples
    'load_folder_file': ('./saved_model', 'checkpoint_24.pth.tar'), 
})

# Plays arenaCompare games in which player1 starts arenaCompare/2 games and player2 starts arenaCompare/2 games.
assert args.arenaCompare % 2 == 0

# make sure the tasks can be split evenly among different remote actors
assert args.numEps % args.actors_num == 0
assert (args.arenaCompare // 2) % args.actors_num == 0 
assert 1000 % args.actors_num == 0 # there are 1000 boards state in test_dataset


def main():
    game = Connect4Game()

    c = Coach(game, args)

    if args.load_model:
        logger.info('Loading checkpoint "%s/%s"...', args.load_folder_file)
        c.loadModel()
        logger.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    c.learn()


if __name__ == "__main__":
    main()
