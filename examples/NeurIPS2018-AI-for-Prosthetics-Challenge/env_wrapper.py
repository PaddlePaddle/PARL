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

import abc
import gym
import math
import numpy as np
import random
from collections import OrderedDict
from osim.env import ProstheticsEnv
from parl.utils import logger
from tqdm import tqdm

MAXTIME_LIMIT = 1000
ProstheticsEnv.time_limit = MAXTIME_LIMIT
FRAME_SKIP = None


class CustomR2Env(gym.Wrapper):
    """Customized target trajectory here, it support 3 ways currently
        1.fixed_speed, e.g. reset(.., fixed_speed=1.25)
        2.stage , e.g. reset(.., stage=1)
        3.boundary, e.g. reset(.., boundary=True)
    """

    def __init__(self,
                 env,
                 time_limit=MAXTIME_LIMIT,
                 discrete_data=False,
                 discrete_bin=10):
        logger.info("[CustomR2Env]type:{}, time_limit:{}".format(
            type(env), time_limit))
        assert isinstance(env, ProstheticsEnv), type(env)
        gym.Wrapper.__init__(self, env)

        self.env.time_limit = time_limit
        self.env.spec.timestep_limit = time_limit
        self.time_limit = time_limit

        # boundary flag
        self._generate_boundary_target_flag = True

        self.discrete_data = discrete_data
        self.discrete_bin = discrete_bin

    def rect(self, row):
        r = row[0]
        theta = row[1]
        x = r * math.cos(theta)
        y = 0
        z = r * math.sin(theta)
        return np.array([x, y, z])

    def _generate_boundary_table(self, ):
        possible_combine = [(math.pi / 8, 0.5), (math.pi / 8, -0.5),
                            (-math.pi / 8, 0.5), (-math.pi / 8, -0.5)]
        self._boundary_table = []
        for a in possible_combine:
            for b in possible_combine:
                for c in possible_combine:
                    self._boundary_table.append([a, b, c])
        assert len(self._boundary_table) == 64

    def generate_boundary_target(self, poisson_lambda=300):
        if self._generate_boundary_target_flag == True:
            self._generate_boundary_target_flag = False
            self._generate_boundary_table()
            self._boundary_index = 0
        nsteps = self.time_limit + 1
        velocity = np.zeros(nsteps)
        heading = np.zeros(nsteps)

        velocity[0] = 1.25
        heading[0] = 0
        trajectory = self._boundary_table[self._boundary_index]

        change = np.cumsum(np.random.poisson(poisson_lambda, 10))
        target_change_times = 0
        for i in range(1, nsteps):
            velocity[i] = velocity[i - 1]
            heading[i] = heading[i - 1]
            if i in change:
                velocity[i] += trajectory[target_change_times][1]
                heading[i] += trajectory[target_change_times][0]
                # trajectory has length 3, the target_change_times should not be large than 2
                target_change_times = min(2, target_change_times + 1)

        self._boundary_index = (self._boundary_index + 1) % 64

    def _generate_target_vel(self, stage, change_num):
        target_vels = None
        if stage == 0:
            target_vels = [1.25]
        elif stage == 1:
            assert change_num >= 1
            interval = 1.0 / self.discrete_bin
            discrete_id = np.random.randint(self.discrete_bin)

            min_vel = 0.75 + discrete_id * interval
            max_vel = 0.75 + (discrete_id + 1) * interval

            target_vels = [1.25]
            for i in range(change_num):
                if i == 0:
                    target_vels.append(random.uniform(min_vel, max_vel))
                else:
                    target_vels.append(target_vels[-1] +
                                       random.uniform(-0.5, 0.5))
        elif stage == 2:
            assert change_num >= 2
            interval = 2.0 / self.discrete_bin
            discrete_id = np.random.randint(self.discrete_bin)
            min_vel = 0.25 + discrete_id * interval
            max_vel = 0.25 + (discrete_id + 1) * interval
            while True:
                target_vels = [1.25]
                for i in range(change_num):
                    target_vels.append(target_vels[-1] +
                                       random.uniform(-0.5, 0.5))
                if target_vels[2] >= min_vel and target_vels[2] <= max_vel:
                    break
        elif stage == 3:
            assert change_num >= 3
            interval = 3.0 / self.discrete_bin
            discrete_id = np.random.randint(self.discrete_bin)
            min_vel = -0.25 + discrete_id * interval
            max_vel = -0.25 + (discrete_id + 1) * interval
            while True:
                target_vels = [1.25]
                for i in range(change_num):
                    target_vels.append(target_vels[-1] +
                                       random.uniform(-0.5, 0.5))
                if target_vels[3] >= min_vel and target_vels[3] <= max_vel:
                    break
        else:
            raise NotImplemented
        logger.info('[CustomR2Env] stage: {}, target_vels: {}'.format(
            stage, target_vels))
        return target_vels

    def generate_stage_targets(self, poisson_lambda=300, stage=None):
        nsteps = self.time_limit + 1
        velocity = np.zeros(nsteps)
        heading = np.zeros(nsteps)

        velocity[0] = 1.25
        heading[0] = 0

        change = np.cumsum(np.random.poisson(poisson_lambda, 10))
        if stage == 0:
            change = []
        elif stage == 1:
            change = change[:1]
        elif stage == 2:
            change = change[:2]
        elif stage == 3:
            if change[3] <= 1000:
                change = change[:4]
            else:
                change = change[:3]
        else:
            raise NotImplemented

        if self.discrete_data:
            target_vels = self._generate_target_vel(
                stage=stage, change_num=len(change))

        change_cnt = 0
        for i in range(1, nsteps):
            velocity[i] = velocity[i - 1]
            heading[i] = heading[i - 1]

            if i in change:
                change_cnt += 1
                if self.discrete_data:
                    velocity[i] = target_vels[change_cnt]
                else:
                    velocity[i] += random.choice([-1, 1]) * random.uniform(
                        -0.5, 0.5)
                heading[i] += random.choice([-1, 1]) * random.uniform(
                    -math.pi / 8, math.pi / 8)

        trajectory_polar = np.vstack((velocity, heading)).transpose()
        targets = np.apply_along_axis(self.rect, 1, trajectory_polar)
        return targets

    def reset(self, **kwargs):
        fixed_speed = None
        if 'fixed_speed' in kwargs:
            fixed_speed = kwargs.pop('fixed_speed', None)
        stage = None
        if 'stage' in kwargs:
            stage = kwargs.pop('stage', None)
        boundary = None
        if 'boundary' in kwargs:
            boundary = kwargs.pop('boundary', None)
        _ = self.env.reset(**kwargs)
        if fixed_speed is not None:
            targets = np.zeros([self.time_limit + 1, 3], dtype=np.float32)
            targets[:, 0] = fixed_speed
            self.env.targets = targets
        elif stage is not None:
            self.env.targets = self.generate_stage_targets(stage=stage)
        elif boundary is not None:
            self.generate_boundary_target()
        else:
            # generate new target
            self.env.generate_new_targets(
                poisson_lambda=int(self.time_limit * (300 / 1000)))
        if 'project' in kwargs:
            if kwargs.get('project') == True:
                return self.env.get_observation()
        return self.env.get_state_desc()

    def step(self, action, **kwargs):
        return self.env.step(action, **kwargs)


def calc_vel_diff(state_desc):
    cur_vel_x = state_desc['body_vel']['pelvis'][0]
    cur_vel_z = state_desc['body_vel']['pelvis'][2]
    target_vel_x = state_desc['target_vel'][0]
    target_vel_z = state_desc['target_vel'][2]
    diff_vel_x = cur_vel_x - target_vel_x
    diff_vel_z = cur_vel_z - target_vel_z

    cur_vel = (cur_vel_x**2 + cur_vel_z**2)**0.5
    target_vel = (target_vel_x**2 + target_vel_z**2)**0.5
    diff_vel = cur_vel - target_vel

    target_theta = math.atan(-1.0 * target_vel_z / target_vel_x)
    # alone y axis
    cur_theta = state_desc['body_pos_rot']['pelvis'][1]
    diff_theta = cur_theta - target_theta

    return cur_vel_x, cur_vel_z, diff_vel_x, diff_vel_z, diff_vel, diff_theta


class ActionScale(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def step(self, action, **kwargs):
        action = (np.copy(action) + 1.0) * 0.5
        action = np.clip(action, 0.0, 1.0)
        return self.env.step(action, **kwargs)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class FrameSkip(gym.Wrapper):
    def __init__(self, env, k):
        logger.info("[FrameSkip]type:{}".format(type(env)))
        gym.Wrapper.__init__(self, env)
        self.frame_skip = k
        global FRAME_SKIP
        FRAME_SKIP = k
        self.frame_count = 0

    def step(self, action, **kwargs):
        r = 0.0
        merge_info = {}
        for k in range(self.frame_skip):
            self.frame_count += 1
            obs, reward, done, info = self.env.step(action, **kwargs)
            r += reward

            for key in info.keys():
                if 'reward' in key:
                    # to assure that we don't igonre other reward
                    # if new reward was added, consider its logic here
                    assert (key == 'shaping_reward') or (
                        key == 'r2_reward') or (key == 'x_offset_reward')
                    merge_info[key] = merge_info.get(key, 0.0) + info[key]
                else:
                    merge_info[key] = info[key]

            if info['target_changed']:
                logger.warn("[FrameSkip] early break since target was changed")
                break

            if done:
                break
        merge_info['frame_count'] = self.frame_count
        return obs, r, done, merge_info

    def reset(self, **kwargs):
        self.frame_count = 0
        return self.env.reset(**kwargs)


class RewardShaping(gym.Wrapper):
    """ A wrapper for reward shaping, note this wrapper must be the first wrapper """

    def __init__(self, env):
        logger.info("[RewardShaping]type:{}".format(type(env)))
        assert (isinstance(env, ProstheticsEnv)
                or isinstance(env, CustomR2Env)), type(env)

        self.step_count = 0
        self.pre_state_desc = None
        self.last_target_vel = None
        self.last_target_change_step = 0
        self.target_change_times = 0
        gym.Wrapper.__init__(self, env)

    @abc.abstractmethod
    def reward_shaping(self, state_desc, reward, done, action):
        """define your own reward computation function
    Args:
        state_desc(dict): state description for current model
        reward(scalar): generic reward generated by env
        done(bool): generic done flag generated by env
    """
        pass

    def step(self, action, **kwargs):
        self.step_count += 1
        obs, r, done, info = self.env.step(action, **kwargs)
        info = self.reward_shaping(obs, r, done, action)
        #logger.info('Step {}: target_vel: {}'.format(self.step_count, obs['target_vel']))
        delta = 0
        if self.last_target_vel is not None:
            delta = np.absolute(
                np.array(self.last_target_vel) - np.array(obs['target_vel']))
        if (self.last_target_vel is None) or np.all(delta < 1e-5):
            info['target_changed'] = False
        else:
            info['target_changed'] = True
            logger.info("[env_wrapper] target_changed, vx:{}   vz:{}".format(
                obs['target_vel'][0], obs['target_vel'][2]))
            self.last_target_change_step = self.step_count
            self.target_change_times += 1
        info['target_change_times'] = self.target_change_times
        self.last_target_vel = obs['target_vel']

        assert 'shaping_reward' in info
        timeout = False
        if self.step_count >= MAXTIME_LIMIT:
            timeout = True

        info['timeout'] = timeout
        self.pre_state_desc = obs
        return obs, r, done, info

    def reset(self, **kwargs):
        self.step_count = 0
        self.last_target_vel = None
        self.last_target_change_step = 0
        self.target_change_times = 0
        obs = self.env.reset(**kwargs)
        self.pre_state_desc = obs
        return obs


class TestReward(RewardShaping):
    """ Reward shaping wrapper for test"""

    def __init__(self, env):
        RewardShaping.__init__(self, env)

    def reward_shaping(self, state_desc, r2_reward, done, action):
        return {'shaping_reward': 0}


class RunFastestReward(RewardShaping):
    """ Reward shaping wrapper for fixed target speed"""

    def __init__(self, env):
        RewardShaping.__init__(self, env)

    def reward_shaping(self, state_desc, r2_reward, done, action):
        if self.pre_state_desc is None:
            x_offset = 0
        else:
            x_offset = state_desc["body_pos"]["pelvis"][
                0] - self.pre_state_desc["body_pos"]["pelvis"][0]

        ret_r = 0
        if self.pre_state_desc is not None:
            l_foot_reward = state_desc["body_pos"]["tibia_l"][
                0] - self.pre_state_desc["body_pos"]["tibia_l"][0]
            r_foot_reward = state_desc["body_pos"]["pros_tibia_r"][
                0] - self.pre_state_desc["body_pos"]["pros_tibia_r"][0]
            ret_r = max(l_foot_reward, r_foot_reward)

        # penalty
        headx = state_desc['body_pos']['head'][0]
        px = state_desc['body_pos']['pelvis'][0]
        headz = state_desc['body_pos']['head'][2]
        pz = state_desc['body_pos']['pelvis'][2]
        kneer = state_desc['joint_pos']['knee_r'][-1]
        kneel = state_desc['joint_pos']['knee_l'][-1]
        lean_x = min(0.3, max(0, px - headx - 0.15)) * 0.05
        lean_z = min(0.3, max(0, pz - headz - 0.15)) * 0.05
        joint = sum([max(0, k - 0.1) for k in [kneer, kneel]]) * 0.03
        penalty = lean_x + lean_z + joint

        ret_r -= penalty * 0.15

        cur_vel_x = state_desc['body_vel']['pelvis'][0]
        cur_vel_z = state_desc['body_vel']['pelvis'][2]
        scalar_vel = math.sqrt(cur_vel_z**2 + cur_vel_x**2)

        info = {
            'shaping_reward': ret_r,
            'r2_reward': r2_reward,
            'x_offset_reward': x_offset,
            'scalar_vel': scalar_vel,
            'mean_action_l2_penalty': 0,
        }
        return info


class FixedTargetSpeedReward(RewardShaping):
    """ Reward shaping wrapper for fixed target speed"""

    def __init__(self, env, target_v, act_penalty_lowerbound,
                 act_penalty_coeff, vel_penalty_coeff):
        RewardShaping.__init__(self, env)

        assert target_v is not None
        assert act_penalty_lowerbound is not None
        assert act_penalty_coeff is not None
        assert vel_penalty_coeff is not None

        self.target_v = target_v
        self.act_penalty_lowerbound = act_penalty_lowerbound
        self.act_penalty_coeff = act_penalty_coeff
        self.vel_penalty_coeff = vel_penalty_coeff

    def reward_shaping(self, state_desc, r2_reward, done, action):
        if self.pre_state_desc is None:
            x_offset = 0
        else:
            x_offset = state_desc["body_pos"]["pelvis"][
                0] - self.pre_state_desc["body_pos"]["pelvis"][0]

        # Reward for not falling
        ret_r = 36

        vel_penalty = ((state_desc["body_vel"]["pelvis"][0] - self.target_v)**2
                       + (state_desc["body_vel"]["pelvis"][2] - 0)**2)

        origin_action_l2_penalty = np.linalg.norm(action)
        action_l2_penalty = max(self.act_penalty_lowerbound,
                                origin_action_l2_penalty)

        ret_r = ret_r - vel_penalty * self.vel_penalty_coeff - action_l2_penalty * self.act_penalty_coeff

        cur_vel_x = state_desc['body_vel']['pelvis'][0]
        cur_vel_z = state_desc['body_vel']['pelvis'][2]
        scalar_vel = math.sqrt(cur_vel_z**2 + cur_vel_x**2)

        info = {
            'shaping_reward': ret_r,
            'r2_reward': r2_reward,
            'x_offset_reward': x_offset,
            'scalar_vel': scalar_vel,
            'mean_action_l2_penalty': origin_action_l2_penalty,
        }
        return info


class Round2Reward(RewardShaping):
    """ Reward shaping wrapper for fixed target speed"""

    def __init__(self, env, act_penalty_lowerbound, act_penalty_coeff,
                 vel_penalty_coeff):
        RewardShaping.__init__(self, env)

        assert act_penalty_lowerbound is not None
        assert act_penalty_coeff is not None
        assert vel_penalty_coeff is not None

        self.act_penalty_lowerbound = act_penalty_lowerbound
        self.act_penalty_coeff = act_penalty_coeff
        self.vel_penalty_coeff = vel_penalty_coeff

    def reward_shaping(self, state_desc, r2_reward, done, action):
        if self.pre_state_desc is None:
            x_offset = 0
        else:
            x_offset = state_desc["body_pos"]["pelvis"][
                0] - self.pre_state_desc["body_pos"]["pelvis"][0]

        # Reward for not falling
        ret_r = 10

        # Small penalty for too much activation (cost of transport)
        muscle_activations = []
        for muscle in sorted(state_desc["muscles"].keys()):
            muscle_activations += [state_desc["muscles"][muscle]["activation"]]
        muscle_penalty = np.sum(np.array(muscle_activations)**2) * 0.001

        vel_penalty = (
            (state_desc["target_vel"][0] - state_desc["body_vel"]["pelvis"][0])
            **2 + (state_desc["target_vel"][2] -
                   state_desc["body_vel"]["pelvis"][2])**2)

        origin_action_l2_penalty = np.linalg.norm(action)
        action_l2_penalty = max(self.act_penalty_lowerbound,
                                origin_action_l2_penalty)

        if self.step_count < 60 or (
                self.step_count - self.last_target_change_step < 60):
            ret_r = ret_r - vel_penalty * self.vel_penalty_coeff
        else:
            ret_r = ret_r - vel_penalty * self.vel_penalty_coeff - action_l2_penalty * self.act_penalty_coeff

        ret_r -= muscle_penalty

        cur_vel_x = state_desc['body_vel']['pelvis'][0]
        cur_vel_z = state_desc['body_vel']['pelvis'][2]
        scalar_vel = math.sqrt(cur_vel_z**2 + cur_vel_x**2)

        info = {
            'shaping_reward': ret_r,
            'r2_reward': r2_reward,
            'x_offset_reward': x_offset,
            'scalar_vel': scalar_vel,
            'mean_action_l2_penalty': origin_action_l2_penalty,
        }
        return info


class ObsTranformerBase(gym.Wrapper):
    def __init__(self, env):
        logger.info("[ObsTranformerBase]type:{}".format(type(env)))
        gym.Wrapper.__init__(self, env)
        self.step_fea = MAXTIME_LIMIT
        global FRAME_SKIP
        self.frame_skip = int(FRAME_SKIP)

    def get_observation(self, state_desc):
        obs = self._get_observation(state_desc)
        if not isinstance(self, PelvisBasedObs):
            cur_vel_x, cur_vel_z, diff_vel_x, diff_vel_z, diff_vel, diff_theta = calc_vel_diff(
                state_desc)
            obs = np.append(obs, [
                cur_vel_x, cur_vel_z, diff_vel_x, diff_vel_z, diff_vel,
                diff_theta
            ])
        else:
            pass
        return obs

    @abc.abstractmethod
    def _get_observation(self, state_desc):
        pass

    def feature_normalize(self, obs, mean, std, duplicate_id):
        scaler_len = mean.shape[0]
        assert obs.shape[0] >= scaler_len
        obs[:scaler_len] = (obs[:scaler_len] - mean) / std
        final_obs = []
        for i in range(obs.shape[0]):
            if i not in duplicate_id:
                final_obs.append(obs[i])
        return np.array(final_obs)

    def step(self, action, **kwargs):
        obs, r, done, info = self.env.step(action, **kwargs)
        if info['target_changed']:
            # reset step_fea when change target
            self.step_fea = MAXTIME_LIMIT

        self.step_fea -= FRAME_SKIP

        obs = self.get_observation(obs)
        return obs, r, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.step_fea = MAXTIME_LIMIT
        obs = self.get_observation(obs)
        return obs


class PelvisBasedObs(ObsTranformerBase):
    def __init__(self, env):
        ObsTranformerBase.__init__(self, env)
        data = np.load('./pelvisBasedObs_scaler.npz')
        self.mean, self.std, self.duplicate_id = data['mean'], data[
            'std'], data['duplicate_id']
        self.duplicate_id = self.duplicate_id.astype(np.int32).tolist()

    def get_core_matrix(self, yaw):
        core_matrix = np.zeros(shape=(3, 3))
        core_matrix[0][0] = math.cos(yaw)
        core_matrix[0][2] = -1.0 * math.sin(yaw)
        core_matrix[1][1] = 1
        core_matrix[2][0] = math.sin(yaw)
        core_matrix[2][2] = math.cos(yaw)
        return core_matrix

    def _get_observation(self, state_desc):
        o = OrderedDict()
        for body_part in [
                'pelvis', 'femur_r', 'pros_tibia_r', 'pros_foot_r', 'femur_l',
                'tibia_l', 'talus_l', 'calcn_l', 'toes_l', 'torso', 'head'
        ]:
            # position
            o[body_part + '_x'] = state_desc['body_pos'][body_part][0]
            o[body_part + '_y'] = state_desc['body_pos'][body_part][1]
            o[body_part + '_z'] = state_desc['body_pos'][body_part][2]
            # velocity
            o[body_part + '_v_x'] = state_desc["body_vel"][body_part][0]
            o[body_part + '_v_y'] = state_desc["body_vel"][body_part][1]
            o[body_part + '_v_z'] = state_desc["body_vel"][body_part][2]

            o[body_part + '_x_r'] = state_desc["body_pos_rot"][body_part][0]
            o[body_part + '_y_r'] = state_desc["body_pos_rot"][body_part][1]
            o[body_part + '_z_r'] = state_desc["body_pos_rot"][body_part][2]

            o[body_part + '_v_x_r'] = state_desc["body_vel_rot"][body_part][0]
            o[body_part + '_v_y_r'] = state_desc["body_vel_rot"][body_part][1]
            o[body_part + '_v_z_r'] = state_desc["body_vel_rot"][body_part][2]

        for joint in [
                'hip_r', 'knee_r', 'ankle_r', 'hip_l', 'knee_l', 'ankle_l',
                'back'
        ]:
            if 'hip' not in joint:
                o[joint + '_joint_pos'] = state_desc['joint_pos'][joint][0]
                o[joint + '_joint_vel'] = state_desc['joint_vel'][joint][0]
            else:
                for i in range(3):
                    o[joint + '_joint_pos_' +
                      str(i)] = state_desc['joint_pos'][joint][i]
                    o[joint + '_joint_vel_' +
                      str(i)] = state_desc['joint_vel'][joint][i]

        # In NIPS2017, only use activation
        for muscle in sorted(state_desc["muscles"].keys()):
            activation = state_desc["muscles"][muscle]["activation"]
            if isinstance(activation, float):
                activation = [activation]
            for i, val in enumerate(activation):
                o[muscle + '_activation_' + str(i)] = activation[i]

            fiber_length = state_desc["muscles"][muscle]["fiber_length"]
            if isinstance(fiber_length, float):
                fiber_length = [fiber_length]
            for i, val in enumerate(fiber_length):
                o[muscle + '_fiber_length_' + str(i)] = fiber_length[i]

            fiber_velocity = state_desc["muscles"][muscle]["fiber_velocity"]
            if isinstance(fiber_velocity, float):
                fiber_velocity = [fiber_velocity]
            for i, val in enumerate(fiber_velocity):
                o[muscle + '_fiber_velocity_' + str(i)] = fiber_velocity[i]

        # z axis of mass have some problem now, delete it later
        o['mass_x'] = state_desc["misc"]["mass_center_pos"][0]
        o['mass_y'] = state_desc["misc"]["mass_center_pos"][1]
        o['mass_z'] = state_desc["misc"]["mass_center_pos"][2]

        o['mass_v_x'] = state_desc["misc"]["mass_center_vel"][0]
        o['mass_v_y'] = state_desc["misc"]["mass_center_vel"][1]
        o['mass_v_z'] = state_desc["misc"]["mass_center_vel"][2]
        for key in ['talus_l_y', 'toes_l_y']:
            o['touch_indicator_' + key] = np.clip(0.05 - o[key] * 10 + 0.5, 0.,
                                                  1.)
            o['touch_indicator_2_' + key] = np.clip(0.1 - o[key] * 10 + 0.5,
                                                    0., 1.)

        # Tranformer
        core_matrix = self.get_core_matrix(o['pelvis_y_r'])
        pelvis_pos = np.array([o['pelvis_x'], o['pelvis_y'],
                               o['pelvis_z']]).reshape((3, 1))
        pelvis_vel = np.array(
            [o['pelvis_v_x'], o['pelvis_v_y'], o['pelvis_v_z']]).reshape((3,
                                                                          1))
        for body_part in [
                'mass', 'femur_r', 'pros_tibia_r', 'pros_foot_r', 'femur_l',
                'tibia_l', 'talus_l', 'calcn_l', 'toes_l', 'torso', 'head'
        ]:
            # rotation
            if body_part != 'mass':
                o[body_part + '_y_r'] -= o['pelvis_y_r']
                o[body_part + '_v_y_r'] -= o['pelvis_v_y_r']
            # position/velocity
            global_pos = []
            global_vel = []
            for each in ['_x', '_y', '_z']:
                global_pos.append(o[body_part + each])
                global_vel.append(o[body_part + '_v' + each])
            global_pos = np.array(global_pos).reshape((3, 1))
            global_vel = np.array(global_vel).reshape((3, 1))
            pelvis_rel_pos = core_matrix.dot(global_pos - pelvis_pos)
            w = o['pelvis_v_y_r']
            offset = np.array(
                [-w * pelvis_rel_pos[2], 0, w * pelvis_rel_pos[0]])
            pelvis_rel_vel = core_matrix.dot(global_vel - pelvis_vel) + offset
            for i, each in enumerate(['_x', '_y', '_z']):
                o[body_part + each] = pelvis_rel_pos[i][0]
                o[body_part + '_v' + each] = pelvis_rel_vel[i][0]

        for key in ['pelvis_x', 'pelvis_z', 'pelvis_y_r']:
            del o[key]

        current_v = np.array(state_desc['body_vel']['pelvis']).reshape((3, 1))
        pelvis_current_v = core_matrix.dot(current_v)
        o['pelvis_v_x'] = pelvis_current_v[0]
        o['pelvis_v_z'] = pelvis_current_v[2]

        res = np.array(list(o.values()))
        res = self.feature_normalize(
            res, mean=self.mean, std=self.std, duplicate_id=self.duplicate_id)

        feet_dis = ((o['tibia_l_x'] - o['pros_tibia_r_x'])**2 +
                    (o['tibia_l_z'] - o['pros_tibia_r_z'])**2)**0.5
        res = np.append(res, feet_dis)
        remaining_time = (self.step_fea -
                          (MAXTIME_LIMIT / 2.0)) / (MAXTIME_LIMIT / 2.0) * -1.0
        res = np.append(res, remaining_time)

        # target driven
        target_v = np.array(state_desc['target_vel']).reshape((3, 1))
        pelvis_target_v = core_matrix.dot(target_v)
        diff_vel_x = pelvis_target_v[0] - pelvis_current_v[0]
        diff_vel_z = pelvis_target_v[2] - pelvis_current_v[2]
        diff_vel = np.sqrt(pelvis_target_v[0] ** 2 + pelvis_target_v[2] ** 2) - \
                   np.sqrt(pelvis_current_v[0] ** 2 + pelvis_current_v[2] ** 2)

        target_vel_x = target_v[0]
        target_vel_z = target_v[2]
        target_theta = math.atan(-1.0 * target_vel_z / target_vel_x)
        current_theta = state_desc['body_pos_rot']['pelvis'][1]
        diff_theta = target_theta - current_theta
        res = np.append(res, [
            diff_vel_x[0] / 3.0, diff_vel_z[0] / 3.0, diff_vel[0] / 3.0,
            diff_theta / (np.pi * 3 / 8)
        ])

        return res


if __name__ == '__main__':
    from osim.env import ProstheticsEnv

    env = ProstheticsEnv(visualize=False)
    env.change_model(model='3D', difficulty=1, prosthetic=True)
    env = CustomR2Env(env)
    env = RunFastestReward(env)
    env = FrameSkip(env, 4)
    env = ActionScale(env)
    env = PelvisBasedObs(env)
    for i in range(64):
        observation = env.reset(project=False, stage=0)
