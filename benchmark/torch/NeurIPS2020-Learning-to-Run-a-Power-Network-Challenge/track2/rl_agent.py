#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
from copy import deepcopy
from utils import process
from powernet_model import PowerNetModel
from grid2op.Agent import BaseAgent
import torch


class RLAgent(BaseAgent):
    def __init__(self, action_space):
        """Initialize a new agent."""
        BaseAgent.__init__(self, action_space=action_space)

        self.action_space = action_space

        self.actions = []
        actions_vec = np.load("./saved_files/top1000_actions.npz")["actions"]
        for i in range(actions_vec.shape[0]):
            act = action_space.from_vect(actions_vec[i])
            self.actions.append(act)

        self.actions = self.actions[:1000]
        self.act_num = len(self.actions)
        self.sub_ids = np.load('./saved_files/sub_id_info.npz')['sub_ids']
        self.do_nothing_action = action_space({})
        self.origin_ids = range(len(self.actions))

        offset = action_space.n_line
        self.action_to_sub_topo = {}
        for sub_id, sub_elem_num in enumerate(action_space.sub_info):
            self.action_to_sub_topo[sub_id] = (offset, offset + sub_elem_num)
            offset += sub_elem_num
        self.step = 0
        self.powernet_model = PowerNetModel()
        self.to_print_data = []

        self.last_disconnect_step = -100
        self.last_diconnect_line = None
        self.simulation_times = 0

    def load(self, path):
        filename = os.path.join(path, 'model.pth')
        self.powernet_model.load_state_dict(torch.load(filename))

    def save(self, path):
        filename = os.path.join(path, 'model.pth')
        torch.save(self.powernet_model.state_dict(), filename)

    def simulate_do_nothing(self, observation):
        init_to_maintain_lines = np.where((observation.time_next_maintenance>0) \
                              & (observation.time_next_maintenance<9))[0]
        to_check_action = self.do_nothing_action
        to_maintain_lines = []
        for line_id in init_to_maintain_lines:
            if observation.line_status[line_id]:
                to_maintain_lines.append(line_id)
        # we do not disconnect the only line in advance
        if len(to_maintain_lines) == 1:
            rest_step = observation.time_next_maintenance[to_maintain_lines[0]]
            if rest_step > 1:
                to_maintain_lines = []
        else:  # we only maintain the first line in `to_maintain_lines`
            to_maintain_lines = to_maintain_lines[:1]

        if len(to_maintain_lines
               ) != 0 and self.step - self.last_disconnect_step > 3:
            line_status = []
            for line_id in to_maintain_lines:
                line_status.append((line_id, -1))
            to_check_action = self.action_space({
                'set_line_status': line_status
            })

            obs_simulate, reward_simulate, done_simulate, info_simulate = observation.simulate(
                to_check_action)
            observation._obs_env._reset_to_orig_state()
        else:
            obs_simulate, reward_simulate, done_simulate, info_simulate = observation.simulate(
                to_check_action)
            observation._obs_env._reset_to_orig_state()
        return obs_simulate, done_simulate, to_check_action, to_maintain_lines

    def find_unaccessible_pos(self, to_check_action):
        if to_check_action == self.do_nothing_action:
            return []
        lines = to_check_action.as_dict()['set_line_status']['disconnected_id']
        arr = []
        for line_id in lines:
            arr.append((line_id, 1))
        act = self.action_space({
            "set_bus": {
                "lines_ex_id": arr,
                "lines_or_id": arr
            }
        })
        pos = np.where(act._set_topo_vect != 0)[0]
        return pos

    def _predict_rho(self, observation):
        extracted_obs = process(observation)
        obs = torch.tensor(extracted_obs, dtype=torch.float32).view(1, -1)
        predicted_rho = self.powernet_model(
            obs).detach().cpu().numpy().reshape(-1)
        sorted_idx = np.argsort(predicted_rho)
        return sorted_idx.tolist(), predicted_rho

    def avoid_overflow(self, observation, reset_action=None):
        if reset_action is None:
            obs_simulate, done_simulate, to_check_action, to_maintain_lines = self.simulate_do_nothing(
                observation)
        else:
            to_check_action = reset_action
            to_maintain_lines = []
            obs_simulate, reward_simulate, done_simulate, info_simulate = observation.simulate(
                to_check_action)
            observation._obs_env._reset_to_orig_state()

        has_overflow = False
        if observation is not None and not any(np.isnan(observation.rho)):
            has_overflow = any(observation.rho > 1.0) or any(
                obs_simulate.rho > 1.0)

        if not (done_simulate or has_overflow) and (
                to_check_action == self.do_nothing_action):
            return self.do_nothing_action, -1
        if to_check_action != self.do_nothing_action and obs_simulate.rho.max(
        ) < 1.0 and not done_simulate:
            return to_check_action, -1

        # action selection and rerank
        top_idx, pred_rho = self._predict_rho(observation)
        action_selected = [False] * len(self.actions)
        for i in range(80):
            idx = top_idx[i]
            action_selected[idx] = True

        # select_action_by_dis
        overflow_lines = np.where(observation.rho > 1.0)[0].tolist()
        if len(overflow_lines) == 0:
            overflow_lines = np.where(obs_simulate.rho > 1.0)[0].tolist()

        best_idx = -1
        least_overflow_action = self.do_nothing_action
        least_overflow = 10.0
        least_obs_simulate = obs_simulate
        if obs_simulate is not None and not any(np.isnan(obs_simulate.rho)):
            least_overflow = float(np.max(obs_simulate.rho))

        if reset_action is None:
            illegal_pos = self.find_unaccessible_pos(to_check_action)
        else:
            illegal_pos = []

        self.simulation_times += 1
        found = False
        for idx in range(self.act_num):
            if not action_selected[idx]: continue
            to_simulate_action = self.actions[idx]
            # check conflict
            if to_check_action != self.do_nothing_action:
                illegal_pos_value = to_simulate_action._set_topo_vect[
                    illegal_pos]
                if np.any(illegal_pos_value):
                    continue
                action1_vec = to_simulate_action.to_vect()
                action2_vec = to_check_action.to_vect()
                to_simulate_action = self.action_space.from_vect(action1_vec +
                                                                 action2_vec)
            legal_action = self.correct_action(observation, to_simulate_action,
                                               self.sub_ids[idx])
            if legal_action == self.do_nothing_action:
                continue

            obs_simulate, reward_simulate, done_simulate, info_simulate = observation.simulate(
                legal_action)
            observation._obs_env._reset_to_orig_state()
            max_rho = obs_simulate.rho.max()

            assert not info_simulate['is_illegal'] and not info_simulate[
                'is_ambiguous']

            if obs_simulate is not None and not any(
                    np.isnan(obs_simulate.rho)):
                if not done_simulate:
                    overflow_value = float(np.max(obs_simulate.rho))
                    if (not found) and (overflow_value < least_overflow):
                        least_overflow = overflow_value
                        least_overflow_action = legal_action
                        least_obs_simulate = obs_simulate
                        best_idx = idx
                    if least_overflow < 0.95:
                        if not found: pass
                        found = True
                        break
                    continue

        if best_idx != -1:
            least_overflow_action = self.correct_action(
                observation, least_overflow_action, self.sub_ids[best_idx])
            if to_check_action != self.do_nothing_action and least_overflow_action != self.do_nothing_action and reset_action is None:
                self.last_disconnect_step = self.step - 1
                self.last_diconnect_line = to_maintain_lines[0]
            if reset_action is not None:
                pass
            return least_overflow_action, self.sub_ids[best_idx]
        else:
            return self.do_nothing_action, -1

    def correct_action(self, observation, to_simulate_action, sub_id):
        if sub_id != -1:
            if observation.time_before_cooldown_sub[sub_id] != 0:
                legal_action_vec = deepcopy(self.do_nothing_action.to_vect())
                return self.do_nothing_action
            else:
                legal_action_vec = deepcopy(to_simulate_action.to_vect())

            sub_topo = self.sub_topo_dict[sub_id]
            if np.any(sub_topo == -1):  # line disconnected
                start, end = self.action_to_sub_topo[sub_id]
                action_topo = legal_action_vec[start:end].astype(
                    "int")  # reference
                action_topo[np.where(
                    sub_topo == -1)[0]] = 0  # done't change bus=-1
                legal_action_vec[start:end] = action_topo
            legal_action = self.action_space.from_vect(legal_action_vec)

        elif sub_id == -1:
            legal_action = to_simulate_action
        else:  # TODO remove
            legal_action = self.do_nothing_action
        return legal_action

    def act(self, observation, reward, done):
        self.step += 1
        offset = 0
        self.sub_topo_dict = {}
        for sub_id, sub_elem_num in enumerate(observation.sub_info):
            sub_topo = observation.topo_vect[offset:offset + sub_elem_num]
            offset += sub_elem_num
            self.sub_topo_dict[sub_id] = sub_topo

        disconnected = np.where(observation.line_status == False)[0].tolist()
        to_maintain_lines = np.where((observation.time_next_maintenance>0) \
                              & (observation.time_next_maintenance<15))[0]
        to_maintain_lines = to_maintain_lines.tolist()
        if len(disconnected) > 0:
            for line_id in disconnected:
                if observation.time_before_cooldown_line[line_id] == 0 and \
                    line_id not in to_maintain_lines:
                    reset_action = self.action_space({
                        "set_line_status": [(line_id, +1)]
                    })
                    obs_simulate, reward_simulate, done_simulate, info_simulate = observation.simulate(
                        reset_action)
                    observation._obs_env._reset_to_orig_state()
                    if np.max(observation.rho) < 1.0 and np.max(
                            obs_simulate.rho) >= 1.0:
                        continue
                    combined_action, sub_id = self.avoid_overflow(
                        observation, reset_action)
                    return combined_action

        if observation is not None and not any(np.isnan(observation.rho)):
            if np.max(observation.rho) < 0.94 and np.any(
                    observation.topo_vect == 2):
                offset = 0
                for sub_id, sub_elem_num in enumerate(observation.sub_info):
                    sub_topo = self.sub_topo_dict[sub_id]

                    if np.any(
                            sub_topo == 2
                    ) and observation.time_before_cooldown_sub[sub_id] == 0:
                        sub_topo = np.where(sub_topo == 2, 1,
                                            sub_topo)  # bus 2 to bus 1
                        sub_topo = np.where(
                            sub_topo == -1, 0,
                            sub_topo)  # don't do action in bus=-1
                        reconfig_sub = self.action_space({
                            "set_bus": {
                                "substations_id": [(sub_id, sub_topo)]
                            }
                        })

                        obs_simulate, reward_simulate, done_simulate, info_simulate = observation.simulate(
                            reconfig_sub)
                        observation._obs_env._reset_to_orig_state()
                        assert not info_simulate[
                            'is_illegal'] and not info_simulate['is_ambiguous']

                        if not done_simulate and obs_simulate is not None and not any(
                                np.isnan(obs_simulate.rho)):
                            if np.max(obs_simulate.rho) < 0.95:
                                return reconfig_sub
                            else:
                                pass
        action, sub_id = self.avoid_overflow(observation)
        return action
