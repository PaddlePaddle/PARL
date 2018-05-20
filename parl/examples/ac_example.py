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

from parl.framework import ComputationWrapperForTest
from parl.framework import Manager
from parl.framework import SimpleRLAgent
from parl.framework import SimpleOnPolicyHelper

if __name__ == '__main__':
    game_options = {
        "pause_screen": False,
        "window_width": 480,
        "window_height": 480,
        "track_type": "straight",
        "track_width": 20.0,
        "track_length": 100.0,
        "track_radius": 30.0,
        "race_full_manouver": False,
        "random": False,
        "difficulty": "easy",
        "context": 1
    }

    ct_options = {"min_batchsize": 1, "max_batchsize": 2}
    wrapper_creators = {
        'RL': (lambda name: ComputationWrapperForTest(name, options=ct_options))
    }

    helper_creators = {
        'RL': (lambda name,comm: SimpleOnPolicyHelper(name, comm, {"sync_steps": 5}))
    }

    manager = Manager(wrapper_creators, helper_creators)
    num_agent = 10
    num_games = 10
    for _ in range(num_agent):
        agent = SimpleRLAgent("simple_race", game_options, num_games)
        manager.add_agent(agent)

    manager.run()
