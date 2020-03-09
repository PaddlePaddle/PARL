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

from copy import deepcopy
import numpy as np
from utils import discretize, linear_discretize
from rlschool import LiftSim


class BaseWrapper(object):
    def __init__(self, env):
        self.env = env
        self._mansion = env._mansion
        self.mansion_attr = self._mansion.attribute

    @property
    def obs_dim(self):
        if hasattr(self.env, 'obs_dim'):
            return self.env.obs_dim
        else:
            return None

    @property
    def act_dim(self):
        if hasattr(self.env, 'act_dim'):
            return self.env.act_dim
        else:
            return None

    def seed(self, seed=None):
        return self.env.seed(seed)

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


class ObsProcessWrapper(BaseWrapper):
    """Extract features of each elevator in LiftSim env
    """

    def __init__(self, env, hour_distize_num=6):
        super(ObsProcessWrapper, self).__init__(env)
        self.hour_distize_num = hour_distize_num
        self.total_steps = 0

    @property
    def obs_dim(self):
        """
        NOTE:
            Keep obs_dim to the return size of function `_mansion_state_process`
        """
        ele_dim = self.mansion_attr.NumberOfFloor * 3 + 34
        obs_dim = (ele_dim + 1) * self.mansion_attr.ElevatorNumber + \
            self.mansion_attr.NumberOfFloor * 2
        obs_dim += self.hour_distize_num
        return obs_dim

    def reset(self):
        """

        Returns:
            obs(list): [[self.obs_dim]] * mansion_attr.ElevatorNumber, features array of all elevators
        """
        obs = self.env.reset()
        self.total_steps = 0
        obs = self._mansion_state_process(obs)
        return obs

    def step(self, action):
        """
        Returns:
            obs(list): nested list, shape of [mansion_attr.ElevatorNumber, self.obs_dim],
                       features array of all elevators
            reward(int): returned by self.env
            done(bool): returned by self.env
            info(dict): returned by self.env
        """
        obs, reward, done, info = self.env.step(action)
        self.total_steps += 1
        obs = self._mansion_state_process(obs)
        return obs, reward, done, info

    def _mansion_state_process(self, mansion_state):
        """Extract features of env
        """
        ele_features = list()
        for ele_state in mansion_state.ElevatorStates:
            ele_features.append(self._ele_state_process(ele_state))
            max_floor = ele_state.MaximumFloor

        target_floor_binaries_up = [0.0 for i in range(max_floor)]
        target_floor_binaries_down = [0.0 for i in range(max_floor)]
        for floor in mansion_state.RequiringUpwardFloors:
            target_floor_binaries_up[floor - 1] = 1.0
        for floor in mansion_state.RequiringDownwardFloors:
            target_floor_binaries_down[floor - 1] = 1.0
        target_floor_binaries = target_floor_binaries_up + target_floor_binaries_down

        raw_time = self.total_steps * 0.5  # timestep seconds
        time_id = int(raw_time % 86400)
        time_id = time_id // (24 / self.hour_distize_num * 3600)
        time_id_vec = discretize(time_id + 1, self.hour_distize_num, 1,
                                 self.hour_distize_num)

        man_features = list()
        for idx in range(len(mansion_state.ElevatorStates)):
            elevator_id_vec = discretize(idx + 1,
                                         len(mansion_state.ElevatorStates), 1,
                                         len(mansion_state.ElevatorStates))
            idx_array = list(range(len(mansion_state.ElevatorStates)))
            idx_array.remove(idx)
            man_features.append(ele_features[idx])
            for left_idx in idx_array:
                man_features[idx] = man_features[idx] + ele_features[left_idx]
            man_features[idx] = man_features[idx] + \
                elevator_id_vec + target_floor_binaries
            man_features[idx] = man_features[idx] + time_id_vec
        return np.asarray(man_features, dtype='float32')

    def _ele_state_process(self, ele_state):
        """Extract features of elevator
        """
        ele_feature = []

        # add floor information
        ele_feature.extend(
            linear_discretize(ele_state.Floor, ele_state.MaximumFloor, 1.0,
                              ele_state.MaximumFloor))

        # add velocity information
        ele_feature.extend(
            linear_discretize(ele_state.Velocity, 21, -ele_state.MaximumSpeed,
                              ele_state.MaximumSpeed))

        # add door information
        ele_feature.append(ele_state.DoorState)
        ele_feature.append(float(ele_state.DoorIsOpening))
        ele_feature.append(float(ele_state.DoorIsClosing))

        # add direction information
        ele_feature.extend(discretize(ele_state.Direction, 3, -1, 1))

        # add load weight information
        ele_feature.extend(
            linear_discretize(ele_state.LoadWeight / ele_state.MaximumLoad, 5,
                              0.0, 1.0))

        # add other information
        target_floor_binaries = [0.0 for i in range(ele_state.MaximumFloor)]
        for target_floor in ele_state.ReservedTargetFloors:
            target_floor_binaries[target_floor - 1] = 1.0
        ele_feature.extend(target_floor_binaries)

        dispatch_floor_binaries = [
            0.0 for i in range(ele_state.MaximumFloor + 1)
        ]
        dispatch_floor_binaries[ele_state.CurrentDispatchTarget] = 1.0
        ele_feature.extend(dispatch_floor_binaries)
        ele_feature.append(ele_state.DispatchTargetDirection)

        return ele_feature


class ActionProcessWrapper(BaseWrapper):
    def __init__(self, env):
        """Map action id predicted by model to action of LiftSim

        """
        super(ActionProcessWrapper, self).__init__(env)

    @property
    def act_dim(self):
        """ 
        NOTE:
            keep act_dim in line with function `_action_idx_to_action`

        Returns:
            int: NumberOfFloor * (2 directions) + (-1 DispatchTarget) + (0 DispatchTarget)
        """
        return self.mansion_attr.NumberOfFloor * 2 + 2

    def step(self, action):
        """
        Args:
            action(list): action_id of all elevators (length = mansion_attr.ElevatorNumber)
        """
        ele_actions = []
        for action_id in action:
            ele_actions.extend(self._action_idx_to_action(action_id))

        # ele_action: list, formatted action for LiftSim env (length = 2 * mansion_attr.ElevatorNumber)
        return self.env.step(ele_actions)

    def _action_idx_to_action(self, action_idx):
        action_idx = int(action_idx)
        realdim = self.act_dim - 2
        if (action_idx == realdim):
            return (0, 1)  # mapped to DispatchTarget=0
        elif (action_idx == realdim + 1):
            return (-1, 1)  # mapped to DispatchTarget=-1
        action = action_idx
        if (action_idx < realdim / 2):
            direction = 1  # up direction
            action += 1
        else:
            direction = -1  # down direction
            action -= int(realdim / 2)
            action += 1
        return (action, direction)


class RewardWrapper(BaseWrapper):
    def __init__(self, env):
        """Design reward of LiftSim env.
        """
        super(RewardWrapper, self).__init__(env)
        self.ele_num = self.mansion_attr.ElevatorNumber

    def step(self, action):
        """Here we return same reward for each elevator,
        you alos can design different rewards of each elevator.

        Returns:
            obs: returned by self.env
            reward: shaping reward
            done: returned by self.env
            info: returned by self.env
        """
        obs, origin_reward, done, info = self.env.step(action)

        reward = -(30 * info['time_consume'] + 0.01 * info['energy_consume'] +
                   100 * info['given_up_persons']) * 1.0e-3 / self.ele_num

        info['origin_reward'] = origin_reward

        return obs, reward, done, info


class MetricsWrapper(BaseWrapper):
    def __init__(self, env):
        super(MetricsWrapper, self).__init__(env)

        self._total_steps = 0
        self._env_reward_1h = 0
        self._env_reward_24h = 0

        self._num_returned = 0
        self._episode_result = []

    def reset(self):
        self._total_steps = 0
        self._env_reward_1h = 0
        self._env_reward_24h = 0
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._total_steps += 1

        self._env_reward_1h += info['origin_reward']
        self._env_reward_24h += info['origin_reward']

        # Treat 1h in LiftSim env as an episode (1step = 0.5s)
        if self._total_steps % (3600 * 2) == 0:  # 1h
            episode_env_reward_1h = self._env_reward_1h
            self._env_reward_1h = 0

            episode_env_reward_24h = None
            if self._total_steps % (24 * 3600 * 2) == 0:  # 24h
                episode_env_reward_24h = self._env_reward_24h
                self._env_reward_24h = 0

            self._episode_result.append(
                [episode_env_reward_1h, episode_env_reward_24h])

        return obs, reward, done, info

    def next_episode_results(self):
        for i in range(self._num_returned, len(self._episode_result)):
            yield self._episode_result[i]
        self._num_returned = len(self._episode_result)
