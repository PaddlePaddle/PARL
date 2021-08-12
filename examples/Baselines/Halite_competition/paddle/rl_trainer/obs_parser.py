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

import numpy as np
from config import config
from .utils import transform_position

size = config["board_size"]


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
