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

import six
from collections import defaultdict

__all__ = ['VectorEnv']


class VectorEnv(object):
    """ vector of envs to support vector reset and vector step.
    `vector_step` api will automatically reset envs which are done.
    """

    def __init__(self, envs):
        """
        Args:
            envs: List of env
        """
        self.envs = envs
        self.envs_num = len(envs)

    def reset(self):
        """
        Returns:
            List of obs
        """
        return [env.reset() for env in self.envs]

    def step(self, actions):
        """
        Args:
            actions: List or array of action

        Returns:
            obs_batch: List of next obs of envs
            reward_batch: List of return reward of envs 
            done_batch: List of done of envs 
            info_batch: List of info of envs 
        """
        obs_batch, reward_batch, done_batch, info_batch = [], [], [], []
        for env_id in six.moves.range(self.envs_num):
            obs, reward, done, info = self.envs[env_id].step(actions[env_id])

            if done:
                obs = self.envs[env_id].reset()

            obs_batch.append(obs)
            reward_batch.append(reward)
            done_batch.append(done)
            info_batch.append(info)
        return obs_batch, reward_batch, done_batch, info_batch
