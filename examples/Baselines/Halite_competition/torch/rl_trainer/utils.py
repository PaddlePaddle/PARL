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

from zerosum_env.envs.halite.helpers import *
from collections import deque
from config import config
import numpy as np
import random

size = config["board_size"]


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


# transform the position in a widely applied setting like the index used in np.array
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


# check the existence of enemies
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
