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
from multiagent.environment import MultiAgentEnv
from multiagent.scenarios import load


class MPEEnv(object):
    def __init__(self, scenario_name, seed):
        """
        multiagent env wrapper

        Args:
            scenario_name (str): scenario of MultiAgentEnv
            seed (int): random seed
        """
        env_list = [
            'simple_attack', 'simple_adversary', 'simple_crypto',
            'simple_push', 'simple_reference', 'simple_speaker_listener',
            'simple_spread', 'simple_tag', 'simple_world_comm',
            'simple_crypto_display'
        ]
        assert scenario_name in env_list, 'Env {} not found (valid envs include {})'.format(
            scenario_name, env_list)
        scenario = load(scenario_name + ".py").Scenario()
        world = scenario.make_world()
        self.env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                                 scenario.observation, scenario.info)
        self.env.seed(seed)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        """
        Returns:
            obs of single env
        """
        return self.env.reset()

    def step(self, actions):
        """
        Args:
            actions: array of action

        Returns:
            obs: array of next obs of env
            reward: array of return reward of env
            done: array of done of env
            info: array of info of env
        """
        agents_actions_list = []
        for agent_id in range(len(self.observation_space)):
            if self.action_space[
                    agent_id].__class__.__name__ == 'MultiDiscrete':
                for action_id in range(self.action_space[agent_id].shape):
                    action_label = actions[agent_id][action_id]
                    max_action = self.action_space[agent_id].high[action_id] + 1
                    action_one_hot = np.squeeze(
                        np.eye(max_action)[action_label])
                    if action_id == 0:
                        action_env = action_one_hot
                    else:
                        action_env = np.concatenate((action_env,
                                                     action_one_hot))
                agents_actions_list.append(action_env)
            else:
                action_label = actions[agent_id]
                max_action = self.action_space[agent_id].n
                action_one_hot = np.squeeze(np.eye(max_action)[action_label])
                agents_actions_list.append(action_one_hot)

        results = self.env.step(agents_actions_list)
        obs, rews, dones, infos = map(np.array, results)

        return obs, rews, dones, infos

    def close(self):
        """close env
        """
        self.env.close()


class VectorEnv(object):
    closed = False
    viewer = None

    def __init__(self, scenario_name, env_num, seed):
        """
        Creates a simple vectorized wrapper for multiple environments

        Args:
            scenario_name (str): scenario of MultiAgentEnv
            env_num (int): number of parallel envs to train
            seed (int): random seed
        """
        self.envs = [
            MPEEnv(scenario_name, seed + rank * 1000)
            for rank in range(env_num)
        ]
        self.raw_env = self.envs[0].env
        self.share_observation_space = self.raw_env.share_observation_space
        self.observation_space = self.raw_env.observation_space
        self.action_space = self.raw_env.action_space
        self.num_envs = env_num

    def reset(self):
        """
        Returns:
            List of obs
        """
        obs = [env.reset() for env in self.envs]
        return np.array(obs)

    def step(self, actions):
        """
        Args:
            actions: array of action

        Returns:
            obs_batch: array of next obs of envs
            reward_batch: array of return reward of envs
            done_batch: array of done of envs
            info_batch: array of info of envs
        """
        obs_batch, rews_batch, dones_batch, infos_batch = [], [], [], []

        for env_id in range(len(actions)):
            obs, rews, dones, infos = self.envs[env_id].step(actions[env_id])

            if 'bool' in dones.__class__.__name__:
                if dones:
                    obs = self.envs[env_id].reset()
            else:
                if np.all(dones):
                    obs = self.envs[env_id].reset()

            obs_batch.append(obs)
            rews_batch.append(rews)
            dones_batch.append(dones)
            infos_batch.append(infos)

        obs_batch = np.array(obs_batch)
        rews_batch = np.array(rews_batch)
        dones_batch = np.array(dones_batch)
        infos_batch = np.array(infos_batch)

        return obs_batch, rews_batch, dones_batch, infos_batch

    def close(self):
        """close all envs
        """
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.closed = True
        for env in self.envs:
            env.close()
