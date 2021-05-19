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

import json
import numpy as np
from connect4_game import Connect4Game


class dotdict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


def win_loss_draw(score):
    if score > 0:
        return 'win'
    if score < 0:
        return 'loss'
    return 'draw'


def get_test_dataset():
    game = Connect4Game()
    test_dataset = []
    with open("refmoves1k_kaggle") as f:
        for line in f:
            data = json.loads(line)

            board = data["board"]
            board = np.reshape(board, game.getBoardSize()).astype(int)
            board[np.where(board == 2)] = -1

            # find out how many moves are played to set the correct mark.
            ply = len([x for x in data["board"] if x > 0])
            if ply & 1:
                player = -1
            else:
                player = 1

            test_dataset.append({
                'board': board,
                'player': player,
                'move_score': data['move score'],
            })
    return test_dataset


"""
split one list to multiple lists
"""
split_group = lambda the_list, group_size: zip(*(iter(the_list), ) * group_size)
