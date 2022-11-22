#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
try:
    import gym
    from gym import spaces
    from pettingzoo.mpe import simple_v2
    from pettingzoo.mpe import simple_adversary_v2
    from pettingzoo.mpe import simple_crypto_v2
    from pettingzoo.mpe import simple_push_v2
    from pettingzoo.mpe import simple_speaker_listener_v3
    from pettingzoo.mpe import simple_spread_v2
    from pettingzoo.mpe import simple_tag_v2
    from pettingzoo.mpe import simple_world_comm_v2
except:
    raise ImportError('Can not use MAenv from parl.env.multiagent_env. \n \
        try `pip install PettingZoo==1.17.0` and `pip install gym==0.23.1` \n \
        (PettingZoo 1.17.0 requires gym>=0.21.0)')

__all__ = ['MAenv']


def MAenv(scenario_name, continuous_actions=False):
    env_list = [
        'simple', 'simple_adversary', 'simple_crypto', 'simple_push',
        'simple_speaker_listener', 'simple_spread', 'simple_tag',
        'simple_world_comm'
    ]
    assert scenario_name in env_list, 'Env {} not found (valid envs include {})'.format(
        scenario_name, env_list)
    if scenario_name == 'simple':
        env = simple_v2.parallel_env(
            max_cycles=25, continuous_actions=continuous_actions)
    elif scenario_name == 'simple_adversary':
        env = simple_adversary_v2.parallel_env(
            N=2, max_cycles=25, continuous_actions=continuous_actions)
    elif scenario_name == 'simple_crypto':
        env = simple_crypto_v2.parallel_env(
            max_cycles=25, continuous_actions=continuous_actions)
    elif scenario_name == 'simple_push':
        env = simple_push_v2.parallel_env(
            max_cycles=25, continuous_actions=continuous_actions)
    elif scenario_name == 'simple_speaker_listener':
        env = simple_speaker_listener_v3.parallel_env(
            max_cycles=25, continuous_actions=continuous_actions)
    elif scenario_name == 'simple_spread':
        env = simple_spread_v2.parallel_env(
            N=3,
            local_ratio=0,
            max_cycles=25,
            continuous_actions=continuous_actions)
    elif scenario_name == 'simple_tag':
        env = simple_tag_v2.parallel_env(
            num_good=1,
            num_adversaries=3,
            num_obstacles=2,
            max_cycles=25,
            continuous_actions=continuous_actions)
    elif scenario_name == 'simple_world_comm':
        env = simple_world_comm_v2.parallel_env(
            num_good=2,
            num_adversaries=4,
            num_obstacles=1,
            num_food=2,
            max_cycles=25,
            num_forests=2,
            continuous_actions=continuous_actions)
    else:
        pass

    env = mpe_wrapper_for_pettingzoo(env, continuous_actions)
    return env


class mpe_wrapper_for_pettingzoo(gym.Wrapper):
    def __init__(self, env=None, continuous_actions=False):
        gym.Wrapper.__init__(self, env)
        self.continuous_actions = continuous_actions
        self.observation_space = list(self.observation_spaces.values())
        self.action_space = list(self.action_spaces.values())
        assert len(self.observation_space) == len(self.action_space)
        self.n = len(self.observation_space)
        self.agents_name = list(self.observation_spaces.keys())
        self.obs_shape_n = [
            self.get_shape(self.observation_space[i]) for i in range(self.n)
        ]
        self.act_shape_n = [
            self.get_shape(self.action_space[i]) for i in range(self.n)
        ]

    def get_shape(self, input_space):
        """
        Args:
            input_space: environment space

        Returns:
            space shape
        """
        if (isinstance(input_space, spaces.Box)):
            if (len(input_space.shape) == 1):
                return input_space.shape[0]
            else:
                return input_space.shape
        elif (isinstance(input_space, spaces.Discrete)):
            return input_space.n
        else:
            print('[Error] shape is {}, not Box or Discrete'.format(
                input_space.shape))
            raise NotImplementedError

    def reset(self):
        obs = self.env.reset()
        return list(obs.values())

    def step(self, actions):
        actions_dict = dict()
        for i, act in enumerate(actions):
            agent = self.agents_name[i]
            if self.continuous_actions:
                assert np.all(((act<=1.0 + 1e-3), (act>=-1.0 - 1e-3))), \
                    'the action should be in range [-1.0, 1.0], but got {}'.format(act)
                high = self.action_space[i].high
                low = self.action_space[i].low
                mapped_action = low + (act - (-1.0)) * ((high - low) / 2.0)
                mapped_action = np.clip(mapped_action, low, high)
                actions_dict[agent] = mapped_action
            else:
                actions_dict[agent] = np.argmax(act)
        obs, reward, done, info = self.env.step(actions_dict)
        return list(obs.values()), list(reward.values()), list(
            done.values()), list(info.values())
