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

import parl
import numpy as np
from parl.utils import logger, RemoteGymEnv


# Example 1, Continuous action space
def main():
    """
    Get your localhost:
    run "xparl start --port ****" on env server
    """
    parl.connect('localhost')
    env = RemoteGymEnv(env_name='HalfCheetah-v1')

    # Run an episode with a random policy
    obs, done = env.reset(), False
    total_steps, episode_reward = 0, 0
    while not done:
        total_steps += 1
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        episode_reward += reward
    logger.info('Episode done, total_steps {}, episode_reward {}'.format(total_steps, episode_reward))


# # Example 2, Discrete action space
# def main():
#     parl.connect('localhost')
#     env = RemoteGymEnv(env_name='MountainCar-v0')
#
#     # Run an episode with a random policy
#     obs, done = env.reset(), False
#     total_steps, episode_reward = 0, 0
#     while not done:
#         total_steps += 1
#         action = np.random.choice(env.action_space.n)
#         next_obs, reward, done, info = env.step(action)
#         episode_reward += reward
#     logger.info('Episode done, total_steps {}, episode_reward {}'.format(total_steps, episode_reward))


if __name__ == '__main__':
    main()
