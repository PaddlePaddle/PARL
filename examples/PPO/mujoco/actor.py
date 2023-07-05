#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import gym
import numpy as np
import parl
from mujoco_model import MujocoModel
from mujoco_agent import MujocoAgent
from parl.algorithms import PPO_Mujoco
from parl.env.compat_wrappers import CompatWrapper


@parl.remote_class(wait=False)
class Actor(object):
    def __init__(self, config, seed=None):
        env = gym.make(config['env'])
        self.env = CompatWrapper(env)
        try:
            self.env.seed(seed)
        except:
            pass

        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        obs_dim += 1  # add 1 to obs dim for time step feature

        model = MujocoModel(obs_dim, act_dim)
        alg = PPO_Mujoco(model, act_dim=act_dim)
        self.agent = MujocoAgent(alg, config)

    def run_episode(self, scaler):
        obs = self.env.reset()
        observes, actions, rewards, unscaled_obs = [], [], [], []
        dones = []
        step = 0.0
        scale, offset = scaler.get()
        scale[-1] = 1.0  # don't scale time step feature
        offset[-1] = 0.0  # don't offset time step feature
        while True:
            obs = obs.reshape((1, -1))
            obs = np.append(obs, [[step]], axis=1)  # add time step feature
            unscaled_obs.append(obs)
            obs = (obs - offset) * scale  # center and scale observations
            obs = obs.astype('float32')
            observes.append(obs)

            action = self.agent.sample(obs)

            action = action.reshape((1, -1)).astype('float32')
            actions.append(action)

            obs, reward, done, _ = self.env.step(np.squeeze(action))
            dones.append(done)
            rewards.append(reward)
            step += 1e-3  # increment time step feature

            if done:
                break
        return {
            'obs': np.concatenate(observes),
            'actions': np.concatenate(actions),
            'rewards': np.array(rewards, dtype='float32'),
            'dones': np.array(dones, dtype='float32'),
            'unscaled_obs': np.concatenate(unscaled_obs)
        }

    def set_weights(self, params):
        self.agent.set_weights(params)
