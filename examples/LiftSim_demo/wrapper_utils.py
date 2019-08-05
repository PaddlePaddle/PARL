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

import sys
import random
import numpy as np
from rlschool import ElevatorState, ElevatorAction
from rlschool import MansionAttribute, MansionState
from rlschool import EPSILON, HUGE
from rlschool import MansionConfig
from rlschool import MansionManager


def discretize(value, n_dim, min_val, max_val):
    """
    discretize a value into a vector of n_dim dimension 1-hot representation
    with the value below min_val being [1, 0, 0, ..., 0]
    and the value above max_val being [0, 0, ..., 0, 1]
    Args:
        value: the value that needs to be discretized into 1-hot format
        n_dim: number of dimensions
        min_val: minimal value in the result
        man_val: maximum value in the result
    Returns:
        the discretized vector
    """
    assert n_dim > 0
    if (n_dim == 1):
        return [1]
    delta = (max_val - min_val) / float(n_dim - 1)
    active_pos = int((value - min_val) / delta + 0.5)
    active_pos = min(n_dim - 1, active_pos)
    active_pos = max(0, active_pos)
    ret_array = [0 for i in range(n_dim)]
    ret_array[active_pos] = 1.0
    return ret_array


def linear_discretize(value, n_dim, min_val, max_val):
    """
    discretize a value into a vector of n_dim dimensional representation
    with the value below min_val being [1, 0, 0, ..., 0]
    and the value above max_val being [0, 0, ..., 0, 1]
    e.g. if n_dim = 2, min_val = 1.0, max_val = 2.0
      if value  = 1.5 returns [0.5, 0.5], if value = 1.8 returns [0.2, 0.8]
    Args:
        value: the value that needs to be discretized
        n_dim: number of dimensions
        min_val: minimal value in the result
        man_val: maximum value in the result
    Returns:
        the discretized vector
    """
    assert n_dim > 0
    if (n_dim == 1):
        return [1]
    delta = (max_val - min_val) / float(n_dim - 1)
    active_pos = int((value - min_val) / delta + 0.5)
    active_pos = min(n_dim - 2, active_pos)
    active_pos = max(0, active_pos)
    anchor_pt = active_pos * delta + min_val
    if (anchor_pt > value and anchor_pt > min_val + 0.5 * delta):
        anchor_pt -= delta
        active_pos -= 1
    weight = (value - anchor_pt) / delta
    weight = min(1.0, max(0.0, weight))
    ret_array = [0 for i in range(n_dim)]
    ret_array[active_pos] = 1.0 - weight
    ret_array[active_pos + 1] = weight
    return ret_array


def ele_state_preprocessing(ele_state):
    """Process elevator state, make it usable for network
    Args:
        ele_state: ElevatorState, nametuple, defined in rlschool/liftsim/environment/mansion/utils.py
    Returns:    
        ele_feature: list of elevator state
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
        linear_discretize(ele_state.LoadWeight / ele_state.MaximumLoad, 5, 0.0,
                          1.0))

    # add other information
    target_floor_binaries = [0.0 for i in range(ele_state.MaximumFloor)]
    for target_floor in ele_state.ReservedTargetFloors:
        target_floor_binaries[target_floor - 1] = 1.0
    ele_feature.extend(target_floor_binaries)

    dispatch_floor_binaries = [0.0 for i in range(ele_state.MaximumFloor + 1)]
    dispatch_floor_binaries[ele_state.CurrentDispatchTarget] = 1.0
    ele_feature.extend(dispatch_floor_binaries)
    ele_feature.append(ele_state.DispatchTargetDirection)

    return ele_feature


def obs_dim(mansion_attr):
    """Calculate the observation dimension
    Args:
        mansion_attr: MansionAttribute, attribute of mansion_manager
    Returns:
        observation dimension
    """
    assert isinstance(mansion_attr, MansionAttribute)
    ele_dim = mansion_attr.NumberOfFloor * 3 + 34
    obs_dim = (ele_dim + 1) * mansion_attr.ElevatorNumber + \
        mansion_attr.NumberOfFloor * 2
    return obs_dim


def act_dim(mansion_attr):
    """Calculate the action dimension, which is number of floor times 2 plus 2.
    The additional two are for special cases: the elevator stops at once if the new dispatch_target is 0,
    the original dispatch_target does not change if dispatch_target is -1. See implementation in
    method action_idx_to_action below.
    Args:
        mansion_attr: MansionAttribute, attribute of mansion_manager
    Returns:
        action dimension
    """
    assert isinstance(mansion_attr, MansionAttribute)
    return mansion_attr.NumberOfFloor * 2 + 2


def mansion_state_preprocessing(mansion_state):
    """Process mansion_state to make it usable for networks, convert it into a numpy array
    Args:
        mansion_state: namedtuple of mansion state, 
            defined in rlschool/liftsim/environment/mansion/utils.py
    Returns:
        the converted numpy array
    """
    ele_features = list()
    for ele_state in mansion_state.ElevatorStates:
        ele_features.append(ele_state_preprocessing(ele_state))
        max_floor = ele_state.MaximumFloor

    target_floor_binaries_up = [0.0 for i in range(max_floor)]
    target_floor_binaries_down = [0.0 for i in range(max_floor)]
    for floor in mansion_state.RequiringUpwardFloors:
        target_floor_binaries_up[floor - 1] = 1.0
    for floor in mansion_state.RequiringDownwardFloors:
        target_floor_binaries_down[floor - 1] = 1.0
    target_floor_binaries = target_floor_binaries_up + target_floor_binaries_down

    idx = 0
    man_features = list()
    for idx in range(len(mansion_state.ElevatorStates)):
        elevator_id_vec = discretize(idx + 1,
                                     len(mansion_state.ElevatorStates), 1,
                                     len(mansion_state.ElevatorStates))
        idx_array = list(range(len(mansion_state.ElevatorStates)))
        idx_array.remove(idx)
        # random.shuffle(idx_array)
        man_features.append(ele_features[idx])
        for left_idx in idx_array:
            man_features[idx] = man_features[idx] + ele_features[left_idx]
        man_features[idx] = man_features[idx] + \
            elevator_id_vec + target_floor_binaries
    return np.asarray(man_features, dtype='float32')


def action_idx_to_action(action_idx, act_dim):
    """Convert action_inx to action
    Args:
        action_idx: the index needed to be converted
        act_dim: action dimension
    Returns:
        the converted namedtuple
    """
    assert isinstance(action_idx, int)
    assert isinstance(act_dim, int)
    realdim = act_dim - 2
    if (action_idx == realdim):
        return ElevatorAction(0, 1)
    elif (action_idx == realdim + 1):
        return ElevatorAction(-1, 1)
    action = action_idx
    if (action_idx < realdim / 2):
        direction = 1
        action += 1
    else:
        direction = -1
        action -= int(realdim / 2)
        action += 1
    return [action, direction]


def action_to_action_idx(action, act_dim):
    """Convert action to number according to act_dim. 
    Args:
        action: namedtuple defined in rlschool/liftsim/environment/mansion/utils.py
        act_dim: action dimension
    Returns:
        action_idx: the result index
    """
    assert isinstance(action, ElevatorAction)
    assert isinstance(act_dim, int)
    realdim = act_dim - 2
    if (action.TargetFloor == 0):
        return realdim
    elif (action.TargetFloor < 0):
        return realdim + 1
    action_idx = 0
    if (action.DirectionIndicator < 0):
        action_idx += int(realdim / 2)
    action_idx += action.TargetFloor - 1
    return action_idx
