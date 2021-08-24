#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
os.environ['PARL_BACKEND'] = 'torch'

import torch
import random
import numpy as np
from config import config
from rl_trainer.controller import Controller
from parl.utils import logger, tensorboard
from parl.utils.window_stat import WindowStat
from zerosum_env import make, evaluate
from zerosum_env.envs.halite.helpers import *

env_seed = config["seed"]
torch.manual_seed(env_seed)
torch.cuda.manual_seed(env_seed)
torch.cuda.manual_seed_all(env_seed)
np.random.seed(env_seed)
random.seed(env_seed)
'''set up agent
In this training script, we only train the first agent
If you want to train the agent by self-play, you can also set 
        player_list = [None, None]
Then, the environment will return the observation of these two agents
Right now, we only use
        player_list = [None, "random"]
Then, the environment will only return the observation of the first agent
'''
player_list = [None, "random"]

# recoring the index of agents being trained
player_index = [i for i in range(len(player_list)) if player_list[i] == None]
player_num = len(player_list)

# set up agent controller
players = [None] * player_num
for player_ind in player_index:
    players[player_ind] = Controller()


# return action for each ship and shipyard
# this function will be only used in testing
def take_action(observation, configuration):
    board = Board(observation, configuration)
    action = players[0].take_action(board, "predict")
    return action


# function for testing
def test_agent():
    players[0].prepare_test()
    rew, _, _, errors = evaluate(
        "halite",
        agents=[take_action, "random"],
        configuration={"randomSeed": env_seed},
        debug=True)
    rew1, rew2 = rew[0][0], rew[0][1]

    if rew1 is None or rew2 is None:
        # Somthing wrong happends in your program(Sometimes the built-in evaluate function will raise
        #                                           timeout exception if the cpu resource is limited)
        # please check the errors to see the detailed message
        print(errors[0])
        return None, None
    return rew1, rew2


def main():

    # statistic data to be recorded
    total_step = 0
    best_win_rate = 0
    best_test_rew = 0
    win_stat = WindowStat(100)

    # start training
    for episode in range(config["episodes"]):

        # build self-play environment
        '''another way to build an environment without a specific seed
        env = make("halite", configuration={"size": config["board_size"]})
        '''
        env = make(
            "halite",
            configuration={
                "size": config["board_size"],
                "randomSeed": env_seed
            })
        configuration = env.configuration
        env = env.train_selfplay(player_list)

        # initialize observation
        observation = env.reset()
        boards = [Board(obs, configuration) for obs in observation]

        # statistic data to be recorded
        episode_step = 0
        episode_halite = [0] * player_num
        episode_ship_halite = [0] * player_num
        episode_ship_num = [0] * player_num
        episode_shipyard_num = [0] * player_num

        # record the alive status of players
        agent_done = [False] * player_num

        # since we train the agent with a random agent, we also save the episode halite obtained by
        # the random agent and compute a winning rate
        random_agent_halite = 0

        # reset the parameters in player
        for ind, board in zip(player_index, boards):
            players[ind].prepare_train()

        while True:

            total_step += 1
            episode_step += 1

            # record the actions of agents
            actions = [None] * player_num

            # sampling actions
            for ind, board in zip(player_index, boards):

                # compute the action for those alive agents
                if not agent_done[ind]:

                    actions[ind] = players[ind].take_action(board, "sample")

            # fetch data of each training agent
            player_infos, terminal = env.step(actions)
            observation_next = [infos[0] for infos in player_infos]
            rews = [infos[1] for infos in player_infos]
            dones = [infos[2] for infos in player_infos]
            boards_next = [
                Board(obs_next, configuration) for obs_next in observation_next
            ]

            # reset the flag of done
            for ind, done in zip(player_index, dones):
                agent_done[ind] = done

            boards = boards_next

            if terminal:

                # update the final state of ships and shipyards and save data
                for ind, board in zip(player_index, boards):
                    players[ind].update_state(board)
                    episode_ship_halite[ind] = sum(
                        [ship.halite for ship in board.current_player.ships])
                    episode_halite[ind] = board.current_player.halite
                    episode_ship_num[ind] = len(board.current_player.ship_ids)
                    episode_shipyard_num[ind] = len(
                        board.current_player.shipyard_ids)

                random_agent_halite = boards[0].opponents[0].halite
                break

        # train agents
        if len(players[0].ship_buffer):
            value_loss, action_loss, entropy = players[0].train_ship_agent()
            tensorboard.add_scalar("train/ship_value_loss", value_loss,
                                   total_step)
            tensorboard.add_scalar("train/ship_policy_loss", action_loss,
                                   total_step)
            tensorboard.add_scalar("train/ship_entropy", entropy, total_step)

        # snippet of code to test the agent
        '''
        # test agent
        if (episode) % config["test_every_episode"] == 0:
            rew1, rew2 = test_agent()
            if rew1 is not None and rew2 is not None:
                tensorboard.add_scalar("test/player_rew", rew1, episode)
                tensorboard.add_scalar("test/random_rew", rew2, episode)
                # saving model
                if rew1 > best_test_rew:
                    best_test_rew = rew1
                    ship_model_path = os.path.join(config["save_path"], 'test_ship_model_ep_%s_rew_%s.pth' % (episode, rew1))
                    players[0].save(ship_model_path)
        '''

        # print statistic data
        for ind, player in zip(player_index, players):
            ship_rew = np.array(player.ship_rew)
            ship_len = np.array(player.ship_len)
            ship_rew = ship_rew.mean() if len(ship_rew) else None
            ship_len = ship_len.mean() if len(ship_len) else None

            message = "player_id:{0}, ".format(ind)
            if ship_rew is not None:
                message += "ship_rew:{0:.2f}, ".format(ship_rew)
                tensorboard.add_scalar("train/ship_rew", ship_rew, total_step)
            if ship_len is not None:
                message += "ship_len:{0:.2f}, ".format(ship_len)
                tensorboard.add_scalar("train/ship_len", ship_len, total_step)

            if ship_rew is not None or ship_len is not None:
                logger.info(message)

        for player_ind, halite, ship_num, shipyard_num, ship_halite in zip(
                player_index, episode_halite, episode_ship_num,
                episode_shipyard_num, episode_ship_halite):

            win_stat.add(halite > random_agent_halite)

            logger.info(
                "player_id:{0}, episode_halite:{1}, ship_num:{2}, shipyard_num:{3}, ship_halite:{4}"
                .format(player_ind, halite, ship_num, shipyard_num,
                        ship_halite))

            tensorboard.add_scalar(
                "train/player{0}_environment_rew".format(player_ind), halite,
                episode)
            tensorboard.add_scalar(
                "train/player{0}_winning_rate".format(player_ind),
                win_stat.mean, episode)

            # save the best model
            if win_stat.mean > best_win_rate:
                best_win_rate = win_stat.mean
                ship_model_path = os.path.join(
                    config["save_path"], 'player%s_best_ship_model_ep_%s.pth' %
                    (player_ind, episode))
                players[player_ind].save(ship_model_path)

        # save the latest model
        ship_model_path = os.path.join(config["save_path"],
                                       'latest_ship_model.pth')
        players[0].save(ship_model_path)


if __name__ == '__main__':
    main()
