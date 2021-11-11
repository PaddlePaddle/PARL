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

import base64
import pickle
import random
import numpy as np

from zerosum_env.envs.halite.helpers import *
from zerosum_env import make, evaluate
from collections import deque

import parl
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distribution import Categorical

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


class Actor(parl.Model):
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
            nn.Conv2D(
                in_channels=5, out_channels=16, kernel_size=(3, 3), stride=2),
            nn.ReLU(),
            nn.Conv2D(
                in_channels=16, out_channels=16, kernel_size=(3, 3), stride=2),
            nn.ReLU(),
            nn.Conv2D(
                in_channels=16,
                out_channels=16,
                kernel_size=(2, 2),
                stride=1,
            ),
            nn.ReLU(),
        )

    def forward(self, x):

        batch_size = x.shape[0]
        ship_feature = x[:, :self.obs_dim]
        world_feature = x[:, self.obs_dim:].reshape((batch_size, 5, 21, 21))
        world_vector = self.network(world_feature).reshape((batch_size, -1))
        x = F.relu(self.l1(ship_feature))
        y = F.relu(self.l2(world_vector))
        z = F.relu(self.l3(paddle.concat((x, y), 1)))
        out = self.l4(z)
        out = F.softmax(self.l4(z), -1)
        return out

    def predict(self, state):
        """Predict action
        Args:
            state (np.array): representation of current state 
        """

        state_tensor = paddle.to_tensor(state, dtype=paddle.float32)
        action = self(state_tensor).detach().numpy().argmax(1)

        return action

    def sample(self, state):
        """Sampling action
        Args:
            state (np.array): representation of current state 
        """

        batch_size = state.shape[0]
        state_tensor = paddle.to_tensor(state, dtype=paddle.float32)
        with paddle.no_grad():
            action_probs = self(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample([1]).reshape((batch_size, )).numpy()
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
        self.ship_actor.set_state_dict(model)


model = b'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'

player = Controller()
model = base64.b64decode(model)
model = pickle.loads(model)
for name, param in model.items():
    model[name] = paddle.to_tensor(param)
player.restore(model)

# function for testing agent
# Attention! This take_action function should be put at the end of this file
player.prepare_test()


def take_action(observation, configuration):
    board = Board(observation, configuration)
    action = player.take_action(board, "sample")
    return action
