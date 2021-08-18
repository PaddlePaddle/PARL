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
from zerosum_env.envs.halite.helpers import *

from .agent import Agent
from .model import Model
from .algorithm import PPO
from .obs_parser import get_ship_feature
from .replay_memory import ReplayMemory
from .utils import is_alive, check_nearby_ship, nearest_shipyard_position, mahattan_distance
from .policy import do_nothing_policy, move_up_policy, move_down_policy, move_left_policy, move_right_policy,\
                    spawn_policy, return_to_base_policy, mine_policy, convert_policy

# the halite we want the agent to mine
halite = config["num_halite"]

# convert action index to specific policy
ship_policies = {
    0: do_nothing_policy,
    1: move_up_policy,
    2: move_down_policy,
    3: move_left_policy,
    4: move_right_policy
}

shipyard_policies = {0: do_nothing_policy, 1: spawn_policy}


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
        ship_actor = Model(
            obs_dim=config["ship_obs_dim"],
            act_dim=config["ship_act_dim"],
            softmax=True)

        ship_critic = Model(obs_dim=config["ship_obs_dim"], act_dim=1)

        ship_alg = PPO(
            actor=ship_actor,
            critic=ship_critic,
            clip_param=0.3,
            value_loss_coef=config["vf_loss_coef"],
            entropy_coef=config["ent_coef"],
            initial_lr=config["lr"],
            max_grad_norm=5)

        self.ship_agent = Agent(ship_alg)

        self.ship_buffer = ReplayMemory(
            config["ship_max_step"],
            config["ship_obs_dim"] + config["world_dim"])

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
            board (envs.halite.helpers.Board): the environment 
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
        """Obtain the id of ships which are not controlled by rules
        Return:
            ready_ship_id (list) : containing the ship ids
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
            method (str) : sampling action or use greedy action
        Return:
            action (dict) : a dictionary recording the action for each ship and shipyard
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
        """Take action for ships
        Args:
            board (envs.halite.helpers.Board): the environment
            method (str) : sampling action or greedy action, either sample or predict
        Return:
            action (dict) : a dictionary recording the action for each ship
        """

        me = board.current_player

        tmp_halite = me.halite

        # spwaning ships until the number reaches the threshold
        if len(me.ships) < config["num_ships"]:

            tmp = config["num_ships"] - len(me.ships)

            for shipyard in me.shipyards:

                # if there is an opponent shipo nearby, the shipyard can spawn a ship to protect itself
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
            board (envs.halite.helpers.Board): the environment
            method (str) : sampling action or greedy action, either sample or predict
        Return:
            action (dict) : a dictionary recording the action for each ship
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
                values, actions, action_log_probs = self.ship_agent.sample(
                    state)
            else:
                actions = self.ship_agent.predict(state)

        # action for those ships whoe are ready
        for ind, ship_id in enumerate(ready_ship_id):

            ship_index = me.ship_ids.index(ship_id)

            ship = me.ships[ship_index]

            ship_policies[actions[ind]](board, ship)

            # set act
            self.ship_states[ship_id]["act"].append(actions[ind])

            if method == "sample":

                self.ship_states[ship.id]["value"].append(values[ind])

                self.ship_states[ship.id]["log_prob"].append(
                    action_log_probs[ind])

        return me.next_actions

    def update_state(self, board):
        """Update the state of the current player
        Args:
            board (envs.halite.helpers.Board): the environment
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
            board (envs.halite.helpers.Board): the environment
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

                # simply set this flag to a negative number
                self.ship_states[ship_id]["halite"].append(-1)

                # whether it's rule policy
                if ship_state["rule"]:

                    act = ship_state["act"]

                    # set the terminal to be 1 and this means the agent finishes its episode
                    if act in ["mine", "convert", "return_to_base"]:

                        ship_state["terminal"].append(1)

                    continue

                # Attacked by the opponents or ships in current teams
                if ship_state["act"][-1] in [0, 1, 2, 3, 4]:

                    # rew for every time step
                    rew = -1

                    # rew for being destroyed
                    rew += -halite

                    ship_state["rew"].append(rew)

                # Set the flags
                ship_state["value"].append(0)
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

                    # reward for every time step
                    rew = -1

                    # mining halite
                    if ship_state["act"][-1] == 0:
                        old_cell = old_board.cells[old_ship.position]
                        new_cell = board.cells[new_ship.position]
                        delta_cell_halite = old_cell.halite - new_cell.halite
                    else:
                        delta_cell_halite = 0

                    # rew for destroying other ships
                    if new_ship.halite != old_ship.halite + delta_cell_halite:
                        destroyed_ship_halite = new_ship.halite - (
                            old_ship.halite + delta_cell_halite)
                        rew += (destroyed_ship_halite / 2)

                    # this ships controlled by model collect the halite we need
                    if new_ship.halite >= halite:
                        ship_state["rew"].append(rew)
                        ship_state["terminal"].append(1)
                        ship_state["value"].append(0)
                        ship_state["full_episode"] = True

                    else:

                        # the environment ends(the enemy dies)
                        if env_done or return_to_base:
                            rew += (ship_state["halite"][-1] - halite)
                            ship_state["rew"].append(rew)
                            next_obs = get_ship_feature(board, new_ship)
                            ship_state["value"].append(
                                self.ship_agent.value(next_obs).squeeze(1)[0])
                            ship_state["full_episode"] = True
                            ship_state["terminal"].append(0)
                        else:
                            ship_state["rew"].append(rew)
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

                # prepare data for training
                if self.train:

                    obs_batch = np.concatenate(ship_state["obs"])
                    action_batch = np.array(ship_state["act"])
                    reward_batch = np.array(ship_state["rew"])
                    terminal_batch = 1 - np.array(ship_state["terminal"])
                    log_prob_batch = np.array(ship_state["log_prob"])
                    value_batch = np.array(ship_state["value"])

                    return_batch = self.compute_returns(
                        reward_batch, terminal_batch, value_batch[-1],
                        config["gamma"])
                    value_batch = value_batch[:-1]

                    advantages = return_batch - terminal_batch * value_batch
                    return_batch = (return_batch - return_batch.mean()) / (
                        return_batch.std() + 1e-5)
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-5)

                    # add data into buffer
                    self.ship_buffer.append(obs_batch, action_batch,
                                            value_batch, return_batch,
                                            log_prob_batch, advantages)

                    # save statistic data
                    self.ship_rew.append(sum(reward_batch))

                    # if the number of halite collected by this ship is greater than defined number,
                    # then we record the number of steps for analysing
                    if ship_state["halite"][-1] > halite:

                        self.ship_len.append(len(action_batch))

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

    def train_ship_agent(self):
        """Train ship agent
        Return:
            v_loss (float) : td error
            a_loss (float) : loss of actor network
            ent (float) : entropy
        """
        v_loss, a_loss, ent = [], [], []
        buf_len = self.ship_buffer.size()
        batch_size = int(config["batch_size"])
        train_times = int(config["train_times"])

        for _ in range(max(1, int(buf_len * train_times / batch_size))):

            obs, action, value, returns, log_prob, adv = self.ship_buffer.sample_batch(
                batch_size)
            value_loss, action_loss, entropy = self.ship_agent.learn(
                obs, action, value, returns, log_prob, adv)
            v_loss.append(value_loss)
            a_loss.append(action_loss)
            ent.append(entropy)

        # reset the buffer to store new data
        self.ship_buffer.reset()

        return np.mean(v_loss), np.mean(a_loss), np.mean(ent)

    def save(self, ship_model_path):
        """Save model for ship and shipyard agent
        Args:
            ship_model_path (str): the path to save the ship model
        """
        self.ship_agent.save(ship_model_path)

    def restore(self, ship_model_path):
        """Restore model
        Args:
            ship_model_path (str): the path to restore the ship model
        """
        self.ship_agent.restore(ship_model_path)

    def compute_returns(self, reward, terminal, next_value, gamma):
        """Compute returns for training value and policy network
        Args:
            reward (np.array) : reward
            terminal (np.array) : the flag of terminal
            next_value (float) :  the value of the last state 
            gamma (float) : coefficient for computing discount value
        Return:
            returns (np.array): discounted value
        """
        returns = np.ones(reward.shape)
        pre_r_sum = next_value
        for step in range(len(reward) - 1, -1, -1):
            returns[step] = reward[step] + gamma * pre_r_sum * terminal[step]
            pre_r_sum = returns[step]

        return returns
