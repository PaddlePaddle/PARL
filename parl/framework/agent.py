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

from abc import ABCMeta, abstractmethod
from multiprocessing import Process, Value
from py_simulator import Simulator
from parl.common.replay_buffer import Experience
from parl.common.logging import GameLogEntry


class AgentBase(Process):
    __metaclass__ = ABCMeta

    def __init__(self, game_name, game_options, num_games):
        super(AgentBase, self).__init__()
        self.id = -1  # isolated agent, not recognized by the framework yet
        self.game_name = game_name
        self.game = Simulator.create(game_name, game_options)
        self.num_games = num_games
        self.helpers = {}
        self.log_q = []
        self.exit_flag = Value('i', 0)

    def add_helper(self, helper):
        self.helpers[helper.name] = helper

    def set_log_queue(log_q):
        self.log_q = log_q

    @abstractmethod
    def _run_one_episode(self):
        pass

    @abstractmethod
    def _game_over(self, game_status):
        pass

    @abstractmethod
    def _game_success(self, game_status):
        pass

    def run(self):
        for _ in range(self.num_games):
            self._run_one_episode()


class SimpleRLAgent(AgentBase):
    def __init__(self, game_name, game_options, num_games):
        super(SimpleRLAgent, self).__init__(game_name, game_options, num_games)

    def _game_over(self, game_status):
        return game_status != 'alive'

    def _game_success(self, game_status):
        return True

    def _run_one_episode(self):
        # sensor_inputs, (prev_)states and actions are all dict
        self.game.reset_game()
        r = 0
        prev_states = []
        log_entry = GameLogEntry(self.id, self.game_name, 'RL')
        game_status = self.game.game_over()
        while self.exit_flag.value == 0 and not self._game_over(game_status):
            sensor_inputs = self.game.get_state()
            predict_inputs = {
                "sensor_inputs": [sensor_inputs, r],
                "states": prev_states
            }
            pred_ret = self.helpers['RL'].predict(predict_inputs)
            a = {"action": pred_ret["actions"]}
            r = self.game.take_actions(a, 1, False)
            game_status = self.game.game_over()
            log_entry.num_steps += 1
            log_entry.total_reward += r
            data = predict_inputs
            data.update({
                "actions": pred_ret["actions"],
                "game_status": game_status
            })
            self.helpers['RL'].store_data(data)
            prev_states = pred_ret["states"]
        log_entry.success = self._game_success(game_status)
        #self.log_q.put(log_entry)
