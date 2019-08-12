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

# wrapper part modified from
# https://github.com/openai/gym/blob/master/gym/core.py

from rlschool import LiftSim
from wrapper_utils import obs_dim, act_dim, mansion_state_preprocessing
from wrapper_utils import action_idx_to_action


class Wrapper(LiftSim):
    def __init__(self, env):
        self.env = env
        self._mansion = env._mansion
        self.mansion_attr = self._mansion.attribute
        self.elevator_num = self.mansion_attr.ElevatorNumber
        self.observation_space = obs_dim(self.mansion_attr)
        self.action_space = act_dim(self.mansion_attr)
        self.viewer = env.viewer

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(
                "attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)

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


class RewardWrapper(Wrapper):
    pass


class ActionWrapper(Wrapper):
    def reset(self):
        return self.env.reset()

    def step(self, action):
        act = []
        for a in action:
            act.extend(self.action(a, self.action_space))
        return self.env.step(act)

    def action(self, action, action_space):
        return action_idx_to_action(action, action_space)


class ObservationWrapper(Wrapper):
    def reset(self):
        self.env.reset()
        return self.observation(self._mansion.state)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return (self.observation(observation), reward, done, info)

    def observation(self, observation):
        return mansion_state_preprocessing(observation)

    @property
    def state(self):
        return self.observation(self._mansion.state)
