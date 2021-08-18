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

from rl_trainer.controller import Controller
from zerosum_env import evaluate
from zerosum_env.envs.halite.helpers import *

if __name__ == '__main__':

    player = Controller()
    player.restore('./model/latest_ship_model.pth')

    # function for testing agent
    def take_action(observation, configuration):
        board = Board(observation, configuration)
        action = player.take_action(board, "predict")
        return action

    # function for testing
    def test_agent():
        player.prepare_test()
        rew, _, _, _ = evaluate(
            "halite",
            agents=[take_action, "random"],
            configuration={"randomSeed": 123456},
            debug=True)
        return rew[0][0], rew[0][1]

    r1, r2 = test_agent()
    print("agent : {0}, random : {1}".format(r1, r2))
