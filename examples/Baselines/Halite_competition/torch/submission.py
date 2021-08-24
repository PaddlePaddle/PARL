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

import pickle
import base64
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from zerosum_env.envs.halite.helpers import *
from zerosum_env import make, evaluate
from collections import deque

import os
os.environ['PARL_BACKEND'] = 'torch'

config = {

    # configuration for env
    "board_size": 21,

    # configuration for the observation of ships
    "world_dim": 5 * 21 * 21,
    "ship_obs_dim": 6,
    "ship_act_dim": 5,

    # the number of halite we want the ships to obtain
    "num_halite": 100,

    # the maximum number of ships
    "num_ships": 10,
}


# obtain the feature of the whole board
def get_world_feature(board):

    size = board.configuration.size
    me = board.current_player

    ships = np.zeros((1, size, size))
    ship_cargo = np.zeros((1, size, size))
    bases = np.zeros((1, size, size))

    map_halite = np.array(board.observation['halite']).reshape(1, size,
                                                               size) / 50

    for _, ship in board.ships.items():
        ships[0, size - ship.position[1] -
              1, ship.position[0]] = 1 if ship.player_id == me.id else -1
        ship_cargo[0, size - ship.position[1] -
                   1, ship.position[0]] = ship.halite

    for _, yard in board.shipyards.items():
        bases[0, size - yard.position[1] - 1, size - yard.position[0] -
              1] = 1 if yard.player_id == me.id else -1

    ship_cargo /= 50

    return np.concatenate([map_halite, ships, ship_cargo, bases], axis=0)


# obtain the distance of current position to other positions
def get_distance_map(position):

    ship_pos = transform_position(position)
    distance_y = (np.ones((size, size)) * np.arange(21))
    distance_x = distance_y.T
    delta_distance_x = abs(distance_x - ship_pos[0])
    delta_distance_y = abs(distance_y - ship_pos[1])
    offset_distance_x = size - delta_distance_x
    offset_distance_y = size - delta_distance_y
    distance_x = np.where(delta_distance_x < offset_distance_x,
                          delta_distance_x, offset_distance_x)
    distance_y = np.where(delta_distance_y < offset_distance_y,
                          delta_distance_y, offset_distance_y)
    distance_map = distance_x + distance_y

    return distance_map.reshape((1, size, size))


# obatin the feature of a specific ship
def get_ship_feature(board, ship):

    me = board.current_player
    size = board.configuration.size

    ship_features = np.zeros(config["ship_obs_dim"] + config["world_dim"])

    # player halite
    ship_features[0] = me.halite / 50

    # halite distance
    ship_features[1] = (me.halite - board.opponents[0].halite) / 50

    # ship_position
    ship_features[2] = (size - ship.position[1] - 1) / size
    ship_features[3] = ship.position[0] / size

    # ship_cargo
    ship_features[4] = ship.halite / 50

    # current step
    ship_features[5] = board.step / 300

    world_feature = get_world_feature(board)

    distance_map = get_distance_map(ship.position)

    cnn_feature = np.concatenate((world_feature, distance_map), axis=0)

    cnn_feature = cnn_feature.reshape((-1))

    ship_features[6:] = cnn_feature

    return ship_features.reshape((1, -1))


# obtain a new position when given an action and a position
def get_new_position(pos, action):

    tmp = [pos[0], pos[1]]
    if action == ShipAction.UP:
        tmp[0] = (pos[0] - 1 + size) % size
    elif action == ShipAction.DOWN:
        tmp[0] = (pos[0] + 1 + size) % size
    elif action == ShipAction.LEFT:
        tmp[1] = (pos[1] - 1 + size) % size
    elif action == ShipAction.RIGHT:
        tmp[1] = (pos[1] + 1 + size) % size

    return tmp


# transform the position in a widly applied setting like the index used in np.array
def transform_position(pos):

    return (size - pos[1] - 1, pos[0])


# obtain the mahattan distance of two positions
def mahattan_distance(pos1, pos2):

    offset_x = abs(pos1[0] - pos2[0])
    offset_y = abs(pos1[1] - pos2[1])

    return min(offset_x, size - offset_x) + min(offset_y, size - offset_y)


# heading to a specific location
def head_to(board, start_pos, des_pos, ignore_teammates):

    ori_pos = start_pos

    act = nearby_enemy(board, board.cells[start_pos].ship, ignore_teammates)
    if act is not None:
        return act

    start_pos = transform_position(start_pos)
    des_pos = transform_position(des_pos)

    central = int(size / 2)
    offset_x = des_pos[0] - central
    offset_y = des_pos[1] - central

    new_pos = ((start_pos[0] - offset_x + size) % size,
               (start_pos[1] - offset_y + size) % size)

    if new_pos[1] < central and check_pos(board, ori_pos, "right"):
        return ShipAction.RIGHT
    if new_pos[1] > central and check_pos(board, ori_pos, "left"):
        return ShipAction.LEFT
    if new_pos[0] < central and check_pos(board, ori_pos, "down"):
        return ShipAction.DOWN
    if new_pos[0] > central and check_pos(board, ori_pos, "up"):
        return ShipAction.UP

    return None


# check the status of the specific position
# whether there is a ship or shipyard
def check_pos(board, pos, act):

    player_id = board.cells[pos].ship.player_id

    if act == "right":
        new_pos = ((pos[0] + 1) % size, pos[1])
    if act == "left":
        new_pos = ((pos[0] - 1) % size, pos[1])
    if act == "up":
        new_pos = (pos[0], (pos[1] + 1) % size)
    if act == "down":
        new_pos = (pos[0], (pos[1] - 1) % size)

    cell = board.cells[new_pos]
    if (cell.ship) \
         or (cell.shipyard and cell.shipyard.player_id != player_id):
        return False

    return True


# chech the existence of enemies
def nearby_enemy(board, ship, ignore_teammates=False):

    ship_position = transform_position(ship.position)

    if ignore_teammates:
        enemy_positions = [
            transform_position(enemy_ship.position)
            for enemy_ship in board.ships.values() if enemy_ship.id != ship.id
            and enemy_ship.player.id != ship.player.id
        ]
    else:
        enemy_positions = [
            transform_position(enemy_ship.position)
            for enemy_ship in board.ships.values() if enemy_ship.id != ship.id
        ]

    enemy_positions.extend([
        transform_position(enemy_shipyard.position)
        for enemy_shipyard in board.shipyards.values()
        if enemy_shipyard.player.id != ship.player.id
    ])

    if len(enemy_positions):
        original_distance = min([
            mahattan_distance(ship_position, enemy_position)
            for enemy_position in enemy_positions
        ])

        if original_distance < 3:

            actions = [
                ShipAction.UP, ShipAction.DOWN, ShipAction.LEFT,
                ShipAction.RIGHT
            ]

            tmp = []

            for action in actions:
                new_position = get_new_position(ship_position, action)
                new_distance = min([
                    mahattan_distance(new_position, enemy_position)
                    for enemy_position in enemy_positions
                ])
                tmp.append(new_distance - original_distance)

            max_dis = max(tmp)
            index = []
            for direct, value in enumerate(tmp):
                if value == max_dis:
                    index.append(direct)
            index = random.choice(index)
            #index = tmp.index(max(tmp))

            return actions[index]

    return None


# obtain the position of the nearest shipyard for a specific ship
def nearest_shipyard_position(board, ship):

    me = board.current_player

    if len(me.shipyards):

        shipyard_positions = [
            transform_position(shipyard.position) for shipyard in me.shipyards
        ]

        ship_position = transform_position(ship.position)

        distance = [
            mahattan_distance(shipyard_position, ship_position)
            for shipyard_position in shipyard_positions
        ]

        index = distance.index(min(distance))

        return me.shipyards[index].position

    else:

        return ship.position


# use bfs to obtain the position of the nearest halite
def nearest_halite(board, ship):

    halite_map = np.array(board.observation['halite']).reshape((size, size))

    check = np.zeros_like(halite_map)

    pos = transform_position(ship.position)

    q = deque()
    q.append(pos)

    while len(q):

        pos = q.popleft()
        if halite_map[pos[0], pos[1]] > 10:
            cell = board.cells[(pos[1], size - pos[0] - 1)]
            if not cell.ship_id and not cell.shipyard_id:
                return (pos[1], size - pos[0] - 1)
        else:
            for offset in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
                tmp = ((pos[0] + offset[0]) % size,
                       (pos[1] + offset[1]) % size)

                if halite_map[tmp[0], tmp[1]] > 10:
                    cell = board.cells[(tmp[1], size - tmp[0] - 1)]

                    assert cell.halite == halite_map[tmp[0], tmp[1]]
                    if not cell.ship_id and not cell.shipyard_id:

                        return (tmp[1], size - tmp[0] - 1)

                if not check[tmp[0], tmp[1]]:
                    q.append(tmp)
                    check[tmp[0], tmp[1]] = 1

    return (pos[1], size - pos[0] - 1)


# determine the alive status of a single player
def is_alive(step, player):

    if len(player.ship_ids):
        alive = True
    else:
        if not len(player.shipyard_ids) or player.halite < 50:
            alive = False
        else:
            alive = True

    alive = False if step >= 299 else alive

    return alive


# check the ships nearby for a specific shipyard
def check_nearby_ship(board, shipyard, enemy):

    size = board.configuration.size
    shipyard_pos = shipyard.position
    player_id = shipyard.player_id

    for offset in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
        new_pos = ((shipyard_pos[0] + offset[0]) % size,
                   (shipyard_pos[1] + offset[1]) % size)
        new_cell = board.cells[new_pos]
        if enemy and new_cell.ship_id and new_cell.ship.player_id != player_id:
            return True
        if not enemy and new_cell.ship_id and new_cell.ship.player_id == player_id:
            return True

    return False


# convert the ship to shipyard
def convert_policy(board, ship):

    ship.next_action = ShipAction.CONVERT


# deposit the halite to the nearest shipyard
# this policy would avoid ships in its teams
def deposit_policy(board, ship):

    shipyard_position = nearest_shipyard_position(board, ship)

    ship.next_action = head_to(board, ship.position, shipyard_position)


# mine the nearest halite (avoiding the ships in current teams and enemies)
def mine_policy(board, ship):

    nearest_halite_pos = nearest_halite(board, ship)

    ship.next_action = head_to(
        board, ship.position, nearest_halite_pos, ignore_teammates=False)


# return to the shipyard
def return_to_base_policy(board, ship):

    shipyard_position = nearest_shipyard_position(board, ship)

    ship.next_action = head_to(
        board, ship.position, shipyard_position, ignore_teammates=True)


# move upwards
def move_up_policy(board, ship):

    act = nearby_enemy(board, ship)
    if act is not None:
        ship.next_action = act
    else:
        ship.next_action = ShipAction.UP


# move downwards
def move_down_policy(board, ship):

    act = nearby_enemy(board, ship)
    if act is not None:
        ship.next_action = act
    else:
        ship.next_action = ShipAction.DOWN


# move left
def move_left_policy(board, ship):

    act = nearby_enemy(board, ship)
    if act is not None:
        ship.next_action = act
    else:
        ship.next_action = ShipAction.LEFT


# move right
def move_right_policy(board, ship):

    act = nearby_enemy(board, ship)
    if act is not None:
        ship.next_action = act
    else:
        ship.next_action = ShipAction.RIGHT


# spawn a ship
def spawn_policy(board, shipyard):

    shipyard.next_action = ShipyardAction.SPAWN


# do nothing
def do_nothing_policy(board, base):

    base.next_action = None


# convert action index to specific policy
ship_policies = {
    0: do_nothing_policy,
    1: move_up_policy,
    2: move_down_policy,
    3: move_left_policy,
    4: move_right_policy
}
shipyard_policies = {0: do_nothing_policy, 1: spawn_policy}

size = config["board_size"]
# the halite we want the agent to mine
halite = config["num_halite"]


class Actor(nn.Module):
    """ Neural Network to approximate v value.
    Args:
        obs_dim (int): Dimension of observation space.
        act_dim (int): Dimension of action space.
    """

    def __init__(self, obs_dim, act_dim):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.l1 = nn.Linear(obs_dim, 16)
        self.l2 = nn.Linear(144, 24)
        self.l3 = nn.Linear(40, 128)
        self.l4 = nn.Linear(128, act_dim)

        self.network = nn.Sequential(
            nn.Conv2d(
                in_channels=5, out_channels=16, kernel_size=(3, 3), stride=2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16, out_channels=16, kernel_size=(3, 3), stride=2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(2, 2),
                stride=1,
            ),
            nn.ReLU(),
        )

    def forward(self, x):

        batch_size = x.shape[0]

        myself_feature = x[:, :self.obs_dim]
        world_feature = x[:, self.obs_dim:].reshape((batch_size, 5, 21, 21))
        world_vector = self.network(world_feature).view(batch_size, -1)

        x = F.relu(self.l1(myself_feature))
        y = F.relu(self.l2(world_vector))
        z = F.relu(self.l3(torch.cat((x, y), 1)))
        out = F.softmax(self.l4(z), -1)
        return out

    def predict(self, state):
        """Predict action
        Args:
            state (np.array): representation of current state 
        """

        state_tensor = torch.tensor(state, dtype=torch.float32)
        action = self(state_tensor).detach().numpy().argmax(1)

        return action

    def sample(self, state):
        """Sampling action
        Args:
            state (np.array): representation of current state 
        """

        state_tensor = torch.tensor(state, dtype=torch.float32)
        probs = self(state_tensor)
        dist = Categorical(probs)
        action = dist.sample().detach().numpy()
        return action


class Controller:
    """Controller.`
    Keep Track of each ship and shipyard
    """

    def __init__(self):
        """Initialize models
        """

        self.init = False

        self.train = True

        # set up agnet for ships
        self.ship_actor = Actor(
            obs_dim=config["ship_obs_dim"], act_dim=config["ship_act_dim"])

    def prepare_test(self):
        """Reset parameters for testing agents
        """
        self.train = False
        self.init = False
        if hasattr(self, "board"):
            del self.board
        if hasattr(self, "ship_states"):
            del self.ship_states

    def prepare_train(self):
        """Reset parameters for training agents
        """
        self.train = True
        self.init = False
        if hasattr(self, "board"):
            del self.board
        if hasattr(self, "ship_states"):
            del self.ship_states

    def reset(self, board):
        """Reset parameters
        Args:
            board : the environment 
        """
        self.init = True
        self.board = board

        player = board.current_player

        self.ship_states = {}

        for ship_id in player.ship_ids:
            # keep track of the status of each ship
            # obs, act, rew, terminal, value, log_prob are the common data we need to record
            # rule indicates whether this ship is controlled by defined rules
            # full_episode indicates the experience collected by this ship is compelete or not
            # halite saves the number of cargo this ship collects
            self.ship_states[ship_id] = {
                "obs": [],
                "act": [],
                "rew": [],
                "terminal": [],
                "value": [],
                "log_prob": [],
                "rule": False,
                "full_episode": False,
                "halite": []
            }

        self.ship_rew = []
        self.ship_len = []

    def get_ship_id(self):
        """Obtain the id of ships which aren't controlled by rules
        """
        ready_ship_id = []

        for ship_id, ship_state in self.ship_states.items():
            if not ship_state["rule"]:
                ready_ship_id.append(ship_id)

        return ready_ship_id

    def take_action(self, board, method='sample'):
        """Take action
        Args:
            board : environment
            method (str) : sampling action or greedy action
        """

        if self.init:
            self.update_state(board)

        if not hasattr(self, "board"):
            self.reset(board)

        # take action for ships
        self.take_ship_action(board, method)

        # take action for shipyards
        self.take_shipyard_action(board)

        me = board.current_player

        self.board = board

        return me.next_actions

    def take_shipyard_action(self, board):
        """Take action for shipyard
        Args:
            board : environment
        """
        me = board.current_player

        tmp_halite = me.halite

        # spwaning ships until the number reaches the threshold
        if len(me.ships) < config["num_ships"]:

            tmp = config["num_ships"] - len(me.ships)

            for shipyard in me.shipyards:

                # if there is an opponent ship nearby, the shipyard can spawn a ship to protect itself
                if check_nearby_ship(board, shipyard, enemy=True):

                    shipyard_policies[1](board, shipyard)

                else:

                    # if the player has extra halite, the shipyard can spawn more ships
                    if tmp_halite > 50 and shipyard.cell.ship_id is None and not check_nearby_ship(
                            board, shipyard, enemy=False):

                        shipyard_policies[1](board, shipyard)

                        tmp -= 1

                        tmp_halite -= 50

                if tmp == 0:

                    break

        return me.next_actions

    def take_ship_action(self, board, method='sample'):
        """Take action for ships
        Args:
            board : environment
            method : sampling action or greedy action, either sample or predict
        """

        me = board.current_player

        # determine a ship and turn it into a base if we do not have a shipyard
        if len(me.shipyards) == 0:

            convert = False

            for ship in me.ships:

                if ship.halite >= 70:

                    self.ship_states[ship.id]["rule"] = True

                    self.ship_states[ship.id]["act"] = "convert"

                    convert = True

                    break

            # if there is no ships we can turn into shipyard
            # then we utilize the defined rules to mine the halite
            if not convert:

                ship = me.ships[0]

                self.ship_states[ship.id]["rule"] = True

                self.ship_states[ship.id]["act"] = "mine"

        # set rule agent
        for ship in me.ships:

            #current time step
            step = board.step

            if not self.ship_states[ship.id]["rule"]:

                shipyard_pos = nearest_shipyard_position(board, ship)

                if board.cells[shipyard_pos].shipyard_id:

                    dis = step + mahattan_distance(shipyard_pos, ship.position)

                    # already obtained K halite or this episode almostly ends,
                    # we force these ships to return to the base
                    if (dis <= 299 and dis >= 294) or ship.halite >= halite:

                        self.ship_states[ship.id]["rule"] = True

                        self.ship_states[ship.id]["act"] = "return_to_base"

                        return_to_base_policy(board, ship)

            else:

                # take action for these rule ships
                if self.ship_states[ship.id]["act"] == "convert":

                    convert_policy(board, ship)

                if self.ship_states[ship.id]["act"] == "mine":

                    mine_policy(board, ship)

                if self.ship_states[ship.id]["act"] == "return_to_base":

                    return_to_base_policy(board, ship)

        # take actions for those non-rule agent by ppo model
        ready_ship_id = self.get_ship_id()

        # obtain the states of these non-rule agent
        state = np.zeros((len(ready_ship_id),
                          config["ship_obs_dim"] + config["world_dim"]))

        if len(ready_ship_id):

            for ind, ship_id in enumerate(ready_ship_id):

                ship_index = me.ship_ids.index(ship_id)

                ship = me.ships[ship_index]

                self.ship_states[ship_id]["obs"].append(
                    get_ship_feature(board, ship))

                state[ind] = self.ship_states[ship_id]["obs"][-1]

            # take action
            if method == 'sample':
                actions = self.ship_actor.sample(state)
            else:
                actions = self.ship_actor.predict(state)

        # action for those ships who are ready
        for ind, ship_id in enumerate(ready_ship_id):

            ship_index = me.ship_ids.index(ship_id)

            ship = me.ships[ship_index]

            ship_policies[actions[ind]](board, ship)

            # set act
            self.ship_states[ship_id]["act"].append(actions[ind])

        return me.next_actions

    def update_state(self, board):
        """Update state of the current player
        Args:
            board : environment
        """

        me = board.current_player

        # fail the game or not
        alive = [is_alive(board.step, me)]

        # the status of opponents
        terminals = [
            is_alive(board.step, opponent) for opponent in board.opponents
        ]

        alive.extend(terminals)

        # whether the status of the environment is over
        env_done = True if not all(alive) or sum(
            alive) == len(alive) - 1 else False

        self.update_ship_state(board, alive[0], env_done)

    def update_ship_state(self, board, alive, env_done):
        """Update the state of each tracked ship
           Define reward and keep tracking of each ship
        Args:
            board : environment
            alive : the status of current player
            env_done : the status of the environment
        """

        old_board = self.board
        new_board = board

        old_me = old_board.current_player
        new_me = board.current_player

        for ship_id, ship_state in self.ship_states.items():

            # ship loss
            if ship_id not in new_me.ship_ids:

                # whether it's rule policy
                if ship_state["rule"]:

                    act = ship_state["act"]

                    # set the terminal to be 1 and this means the agent finishes its episode
                    if act in ["mine", "convert", "return_to_base"]:

                        ship_state["terminal"].append(1)

                    continue

                # Set the flags
                ship_state["terminal"].append(1)
                ship_state["full_episode"] = True

            else:

                old_ship = old_board.ships[ship_id]

                new_ship = board.ships[ship_id]

                # record the halite of this ship
                self.ship_states[ship_id]["halite"].append(new_ship.halite)

                # whether it's controlled by rules
                # define the status of these rule ships
                if ship_state["rule"]:

                    if ship_state["act"] == "mine":

                        # The agent will only stop mining until the number of halite is more than 70
                        # when it's asked to mine and build a shipyard
                        if new_ship.halite > 70:

                            ship_state["terminal"].append(1)

                    if ship_state["act"] == "return_to_base":

                        # To determine whether the ship is at the shipyard or not
                        new_cell = board.cells[new_ship.position]

                        if new_cell.shipyard_id and new_cell.shipyard.player_id == new_ship.player_id:

                            ship_state["terminal"].append(1)

                    continue

                # To determine whether the ship should return to base or not
                return_to_base = False
                shipyard_pos = nearest_shipyard_position(board, new_ship)
                if board.cells[shipyard_pos].shipyard_id:
                    dis = board.step + mahattan_distance(
                        shipyard_pos, new_ship.position)
                    return_to_base = (dis <= 299 and dis >= 294)

                if ship_state["act"][-1] in [0, 1, 2, 3, 4]:

                    # this ships controlled by model collect the halite we need
                    if new_ship.halite >= halite:
                        ship_state["terminal"].append(1)
                        ship_state["full_episode"] = True

                    else:

                        # the environment ends(the enemy dies)
                        if env_done or return_to_base:
                            ship_state["full_episode"] = True
                            ship_state["terminal"].append(0)
                        else:
                            ship_state["terminal"].append(0)

        # record the id of the ship we want to loss track of
        eliminated_ship_ids = []

        for ship_id, ship_state in self.ship_states.items():

            if ship_state["rule"]:

                # if the rule agent finish its goal, then we should not keep track of this agent.
                if len(ship_state["terminal"]) and ship_state["terminal"][-1]:

                    eliminated_ship_ids.append(ship_id)

                continue

            if ship_state["full_episode"]:

                eliminated_ship_ids.append(ship_id)

        # loss track of eliminated ships
        for ship_id in eliminated_ship_ids:

            del self.ship_states[ship_id]

        # add new ship id (keep track of a new ship)
        for ship_id in new_me.ship_ids:

            if ship_id not in self.ship_states.keys():

                self.ship_states[ship_id] = {
                    "obs": [],
                    "act": [],
                    "rew": [],
                    "terminal": [],
                    "value": [],
                    "log_prob": [],
                    "rule": False,
                    "full_episode": False,
                    "halite": []
                }

    def restore(self, model):
        """Restore model
        """
        self.ship_actor.load_state_dict(model)


model = b'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'

player = Controller()
model = base64.b64decode(model)
model = pickle.loads(model)
for name, param in model.items():
    model[name] = torch.tensor(param)
player.restore(model)

# function for testing agent
# Attention! This take_action function should be put in the end of this file
player.prepare_test()


def take_action(observation, configuration):
    board = Board(observation, configuration)
    action = player.take_action(board, "predict")
    return action
