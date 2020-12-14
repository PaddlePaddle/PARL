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

import numpy as np
import pickle
from grid2op.Agent import BaseAgent
from grid2op.dtypes import dt_int
from copy import deepcopy
from powernet_model import CombinedActionsModel, UnitaryActionModel
from es import ES, EnsembleES
from es_agent import CombineESAgent, UnitaryESAgent


class Track1PowerNetAgent(BaseAgent):
    def __init__(self, action_space):
        BaseAgent.__init__(self, action_space=action_space)
        self.simulate_times = 0

        unitary_action_model = UnitaryActionModel()
        algorithm = ES(unitary_action_model)
        self.unitary_es_agent = UnitaryESAgent(algorithm)

        combined_actions_model_1 = CombinedActionsModel()
        combined_actions_model_2 = CombinedActionsModel()
        ensemble_algorithm = EnsembleES(combined_actions_model_1,
                                        combined_actions_model_2)
        self.combine_es_agent = CombineESAgent(ensemble_algorithm)

        self.unitary_es_agent.restore('./saved_files',
                                      'unitary_action_model.ckpt')
        self.combine_es_agent.restore('./saved_files',
                                      'combined_actions_model.ckpt')

        unitary_actions_vec = np.load(
            "./saved_files/v6_top500_unitary_actions.npz")["actions"]
        self.unitary_actions = []
        for i in range(unitary_actions_vec.shape[0]):
            action = action_space.from_vect(unitary_actions_vec[i])
            self.unitary_actions.append(action)

        redispatch_actions_vec = np.load(
            "./saved_files/redispatch_actions.npz")["actions"]
        self.redispatch_actions = []
        for i in range(redispatch_actions_vec.shape[0]):
            action = action_space.from_vect(redispatch_actions_vec[i])
            self.redispatch_actions.append(action)

        with open("./saved_files/action_to_sub_id.pickle", "rb") as f:
            self.action_to_sub_id = pickle.load(f)

        self.after_line56_or_line45_disconnect_actions = []
        self.three_sub_action_to_sub_ids = {}

        actions_vec = np.load(
            "./saved_files/v10_merge_three_sub_actions.npz")["actions"]
        for i in range(actions_vec.shape[0]):
            action = action_space.from_vect(actions_vec[i])
            self.after_line56_or_line45_disconnect_actions.append(action)

        with open("saved_files/three_sub_action_to_sub_ids.pickle", "rb") as f:
            self.three_sub_action_to_sub_ids = pickle.load(f)

        self.used_combine_actions = False
        self.redispatch_cnt = 0
        self.max_redispatch_cnt = 3
        self.serial_actions = []

        self.do_nothing_action = action_space({})
        self.action_space = action_space

        offset = 59
        self.action_to_sub_topo = {}
        for sub_id, sub_elem_num in enumerate(action_space.sub_info):
            self.action_to_sub_topo[sub_id] = (offset, offset + sub_elem_num)
            offset += sub_elem_num

        self.observation = None

        self.redispatch_months = set([3])

    def act(self, observation, reward, done):
        self.observation = observation

        action = self._first_stage_act()

        if observation.month in self.redispatch_months:
            action = self._try_combine_with_redispatch(observation, action)

        return action

    def _try_combine_with_redispatch(self, observation, action):
        if (observation.line_status[45] == False or observation.line_status[56] == False) \
                and action != self.do_nothing_action \
                and self.redispatch_cnt < self.max_redispatch_cnt \
                and action.impact_on_objects()['topology']['changed']:
            obs_simulate, reward_simulate, done_simulate, info_simulate = observation.simulate(
                action)
            observation._obs_env._reset_to_orig_state()
            assert not info_simulate['is_illegal'] and not info_simulate[
                'is_ambiguous']

            origin_rho = 10.0
            if not done_simulate:
                origin_rho = obs_simulate.rho.max()

            least_rho = origin_rho
            best_action = None
            for redispatch_action in self.redispatch_actions:
                combine_action = self.action_space.from_vect(
                    action.to_vect() + redispatch_action.to_vect())
                obs_simulate, reward_simulate, done_simulate, info_simulate = observation.simulate(
                    combine_action)
                observation._obs_env._reset_to_orig_state()
                assert not info_simulate['is_illegal'] and not info_simulate[
                    'is_ambiguous']

                max_rho = 10.0
                if not done_simulate:
                    max_rho = obs_simulate.rho.max()
                if max_rho < least_rho:
                    least_rho = max_rho
                    best_action = combine_action

            if least_rho < origin_rho:
                action = best_action
                self.redispatch_cnt += 1

        return action

    def _first_stage_act(self):
        self._calc_sub_topo_dict()

        action = self._check_serial_actions()
        if action is not None:
            return action

        action = self._reconnect_action()
        if action is not None:
            return action

        # update global variables
        if np.all(self.observation.topo_vect != -1):
            self.used_combine_actions = False
            self.redispatch_cnt = 0

        if self.observation is not None and not any(
                np.isnan(self.observation.rho)):
            if np.all(self.observation.topo_vect != -1):
                action = self._reset_redispatch()
                if action is not None:
                    return action

                action = self._reset_topology()
                if action is not None:
                    return action

        obs_simulate, reward_simulate, done_simulate, info_simulate = self.observation.simulate(
            self.do_nothing_action)
        self.observation._obs_env._reset_to_orig_state()

        has_overflow = False
        if self.observation is not None and not any(
                np.isnan(self.observation.rho)):
            has_overflow = any(self.observation.rho > 1.0)

        if done_simulate or has_overflow:
            if (self.observation.line_status[56] == False
                    or self.observation.line_status[45] == False
                ) and not self.used_combine_actions:
                action = self._three_sub_action()
                if action is not None:
                    return action

            action = self._unitary_actions_simulate()
            if action is not None:
                return action

        return self.do_nothing_action

    def _three_sub_action(self):
        predicted_rho = self.combine_es_agent.predict(self.observation)
        sorted_idx = np.argsort(predicted_rho)

        sub_ids = []
        for best_idx in sorted_idx:
            best_act = self.after_line56_or_line45_disconnect_actions[best_idx]
            sub_ids = self.three_sub_action_to_sub_ids[best_idx]

            if not np.all(
                    self.observation.time_before_cooldown_sub[sub_ids] == 0):
                sub_ids = []
                continue

            legal_action_vec = deepcopy(best_act.to_vect())
            for sub_id in sub_ids:
                sub_topo = self.sub_toop_dict[sub_id]
                if np.any(sub_topo == -1):  # line disconnected
                    start, end = self.action_to_sub_topo[sub_id]

                    action_topo = legal_action_vec[start:end]
                    action_topo[np.where(
                        sub_topo == -1)[0]] = 0  # done't change bus=-1

                    legal_action_vec[start:end] = action_topo

            best_act = self.action_space.from_vect(legal_action_vec)
            break

        self.used_combine_actions = True

        self.serial_actions = []
        for sub_id in sub_ids:
            start, end = self.action_to_sub_topo[sub_id]

            action_topo = best_act.to_vect()[start:end]

            act = self.action_space({}).to_vect()
            act[start:end] = action_topo

            self.serial_actions.append((self.action_space.from_vect(act),
                                        sub_id))

        action = self._check_serial_actions()

        return action

    def _check_serial_actions(self):
        # serial actions
        if len(self.serial_actions) > 0:
            least_rho = 10.0
            best_idx = -1
            best_act = None
            for i, (act, sub_id) in enumerate(self.serial_actions):

                assert self.observation.time_before_cooldown_sub[sub_id] == 0

                legal_action_vec = deepcopy(act.to_vect())

                sub_topo = self.sub_toop_dict[sub_id]
                if np.any(sub_topo == -1):  # line disconnected
                    start, end = self.action_to_sub_topo[sub_id]

                    action_topo = legal_action_vec[start:end].astype(
                        "int")  # reference
                    action_topo[np.where(
                        sub_topo == -1)[0]] = 0  # done't change bus=-1

                    legal_action_vec[start:end] = action_topo

                legal_action = self.action_space.from_vect(legal_action_vec)

                obs_simulate, reward_simulate, done_simulate, info_simulate = self.observation.simulate(
                    legal_action)
                self.observation._obs_env._reset_to_orig_state()

                assert not info_simulate['is_illegal'] and not info_simulate[
                    'is_ambiguous']

                if done_simulate:
                    continue

                assert not any(np.isnan(obs_simulate.rho))

                max_rho = np.max(obs_simulate.rho)
                if max_rho < least_rho:
                    least_rho = max_rho
                    best_idx = i
                    best_act = legal_action
            if best_idx != -1:
                del self.serial_actions[best_idx]
                return best_act
            else:
                self.serial_actions = []
                return None

    def _unitary_actions_simulate(self):
        self.simulate_times += 1
        best_action = None

        obs_simulate, reward_simulate, done_simulate, info_simulate = self.observation.simulate(
            self.do_nothing_action)
        self.observation._obs_env._reset_to_orig_state()

        least_overflow = 2.0
        if obs_simulate is not None and not any(np.isnan(obs_simulate.rho)):
            least_overflow = float(np.max(obs_simulate.rho))

        predicted_rho = self.unitary_es_agent.predict(self.observation)
        sorted_idx = np.argsort(predicted_rho).tolist()
        top_idx = sorted_idx[:350]
        top_idx.sort()

        for idx in top_idx:
            action = self.unitary_actions[idx]
            sub_id = self.action_to_sub_id[idx]
            if sub_id != "redispatch":  # topology change action
                if self.observation.time_before_cooldown_sub[sub_id] != 0:
                    continue

                legal_action_vec = deepcopy(action.to_vect())

                sub_topo = self.sub_toop_dict[sub_id]
                if np.any(sub_topo == -1):  # line disconnected
                    start, end = self.action_to_sub_topo[sub_id]

                    action_topo = legal_action_vec[start:end].astype(
                        "int")  # reference
                    action_topo[np.where(
                        sub_topo == -1)[0]] = 0  # done't change bus=-1
                    legal_action_vec[start:end] = action_topo

                legal_action = self.action_space.from_vect(legal_action_vec)

            else:
                legal_action = action

            obs_simulate, reward_simulate, done_simulate, info_simulate = self.observation.simulate(
                legal_action)
            self.observation._obs_env._reset_to_orig_state()

            assert not info_simulate['is_illegal'] and not info_simulate[
                'is_ambiguous']

            if obs_simulate is not None and not any(
                    np.isnan(obs_simulate.rho)):
                if not done_simulate:
                    overflow_value = float(np.max(obs_simulate.rho))
                    if overflow_value < least_overflow:
                        least_overflow = overflow_value
                        best_action = legal_action

        return best_action

    def _reset_redispatch(self):
        if np.max(self.observation.rho) < 1.0:
            # reset redispatch
            if not np.all(self.observation.target_dispatch == 0.0):
                gen_ids = np.where(self.observation.gen_redispatchable)[0]
                gen_ramp = self.observation.gen_max_ramp_up[gen_ids]
                changed_idx = np.where(
                    self.observation.target_dispatch[gen_ids] != 0.0)[0]
                redispatchs = []
                for idx in changed_idx:
                    target_value = self.observation.target_dispatch[gen_ids][
                        idx]
                    value = min(abs(target_value), gen_ramp[idx])
                    value = -1 * target_value / abs(target_value) * value
                    redispatchs.append((gen_ids[idx], value))
                act = self.action_space({"redispatch": redispatchs})

                obs_simulate, reward_simulate, done_simulate, info_simulate = self.observation.simulate(
                    act)
                self.observation._obs_env._reset_to_orig_state()

                assert not info_simulate['is_illegal'] and not info_simulate[
                    'is_ambiguous']

                if not done_simulate and obs_simulate is not None and not any(
                        np.isnan(obs_simulate.rho)):
                    if np.max(obs_simulate.rho) < 1.0:
                        return act

    def _reset_topology(self):
        if np.max(self.observation.rho) < 0.95:
            offset = 0
            for sub_id, sub_elem_num in enumerate(self.observation.sub_info):
                sub_topo = self.sub_toop_dict[sub_id]

                if sub_id == 28:
                    sub28_topo = np.array([2.0, 1.0, 2.0, 1.0, 1.0])
                    if not np.all(
                            sub_topo.astype(int) == sub28_topo.astype(int)
                    ) and self.observation.time_before_cooldown_sub[
                            sub_id] == 0:
                        sub_id = 28
                        act = self.action_space({
                            "set_bus": {
                                "substations_id": [(sub_id, sub28_topo)]
                            }
                        })

                        obs_simulate, reward_simulate, done_simulate, info_simulate = self.observation.simulate(
                            act)
                        self.observation._obs_env._reset_to_orig_state()
                        assert not info_simulate[
                            'is_illegal'] and not info_simulate['is_ambiguous']
                        if not done_simulate and obs_simulate is not None and not any(
                                np.isnan(obs_simulate.rho)):
                            if np.max(obs_simulate.rho) < 0.95:
                                return act
                    continue

                if np.any(
                        sub_topo == 2
                ) and self.observation.time_before_cooldown_sub[sub_id] == 0:
                    sub_topo = np.where(sub_topo == 2, 1,
                                        sub_topo)  # bus 2 to bus 1
                    sub_topo = np.where(sub_topo == -1, 0,
                                        sub_topo)  # don't do action in bus=-1
                    reconfig_sub = self.action_space({
                        "set_bus": {
                            "substations_id": [(sub_id, sub_topo)]
                        }
                    })

                    obs_simulate, reward_simulate, done_simulate, info_simulate = self.observation.simulate(
                        reconfig_sub)
                    self.observation._obs_env._reset_to_orig_state()

                    assert not info_simulate[
                        'is_illegal'] and not info_simulate['is_ambiguous']

                    if not done_simulate:
                        assert np.any(
                            obs_simulate.topo_vect !=
                            self.observation.topo_vect)  # have some impact

                    if not done_simulate and obs_simulate is not None and not any(
                            np.isnan(obs_simulate.rho)):
                        if np.max(obs_simulate.rho) < 0.95:
                            return reconfig_sub

        if np.max(self.observation.rho) >= 1.0:
            sub_id = 28
            sub_topo = self.sub_toop_dict[sub_id]
            if np.any(
                    sub_topo == 2
            ) and self.observation.time_before_cooldown_sub[sub_id] == 0:
                sub28_topo = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
                act = self.action_space({
                    "set_bus": {
                        "substations_id": [(sub_id, sub28_topo)]
                    }
                })

                obs_simulate, reward_simulate, done_simulate, info_simulate = self.observation.simulate(
                    act)
                self.observation._obs_env._reset_to_orig_state()
                assert not info_simulate['is_illegal'] and not info_simulate[
                    'is_ambiguous']
                if not done_simulate and obs_simulate is not None and not any(
                        np.isnan(obs_simulate.rho)):
                    if np.max(obs_simulate.rho) < 0.99:
                        return act

    def _calc_sub_topo_dict(self):
        offset = 0
        self.sub_toop_dict = {}
        for sub_id, sub_elem_num in enumerate(self.observation.sub_info):
            sub_topo = self.observation.topo_vect[offset:offset + sub_elem_num]
            offset += sub_elem_num
            self.sub_toop_dict[sub_id] = sub_topo

    def _reconnect_action(self):
        disconnected = np.where(
            self.observation.line_status == False)[0].tolist()

        least_rho = 2.0
        best_action = None
        for line_id in disconnected:
            if self.observation.time_before_cooldown_line[line_id] == 0:
                action = self.action_space({
                    "set_line_status": [(line_id, +1)]
                })
                obs_simulate, reward_simulate, done_simulate, info_simulate = self.observation.simulate(
                    action)
                self.observation._obs_env._reset_to_orig_state()

                if np.max(self.observation.rho) < 1.0 and np.max(
                        obs_simulate.rho) >= 1.0:
                    continue

                return action
