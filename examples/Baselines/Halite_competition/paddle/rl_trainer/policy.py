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
from .utils import *


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
