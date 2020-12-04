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

import grid2op
import lightsim2grid
assert grid2op.__version__ == "1.2.2"
assert lightsim2grid.__version__ == "0.2.4"

from lightsim2grid.LightSimBackend import LightSimBackend
import numpy as np
from agent import Track2PowerNetAgent
import argparse

MAX_TIMESTEP = 7 * 288


class Evaluator(object):
    def __init__(self):
        backend = LightSimBackend()
        env = grid2op.make("l2rpn_neurips_2020_track2_small", backend=backend)

        self.agent = Track2PowerNetAgent(env.action_space)
        self.env = env

    def run(self, num_episodes):
        steps_buffer = []
        rewards_buffer = []

        for _ in range(num_episodes):
            _ = self.env.reset()
            max_day = (
                self.env.chronics_handler.max_timestep() - MAX_TIMESTEP) // 288
            start_timestep = np.random.randint(
                max_day) * 288 - 1  # start at 00:00
            if start_timestep > 0:
                self.env.fast_forward_chronics(start_timestep)

            obs = self.env.get_obs()
            done = False
            steps = 0
            rewards = 0
            while not done:
                action = self.agent.act(obs, None, None)
                obs, reward, done, info = self.env.step(action)
                assert not info['is_illegal'] and not info['is_ambiguous']
                rewards += reward
                steps += 1
                if steps >= MAX_TIMESTEP:
                    break
            steps_buffer.append(steps)
            rewards_buffer.append(rewards)

        return np.mean(steps_buffer), np.mean(rewards_buffer)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=10,
        help='num episodes to evaluate')
    parser.add_argument(
        '--env_dir', default=None, help='directory path of the environment')

    args = parser.parse_args()
    if args.env_dir:
        grid2op.change_local_dir(args.env_dir)

    evaluator = Evaluator()
    mean_steps, mean_rewards = evaluator.run(args.num_episodes)
    print('num_episodes: {}, mean_reward: {:.1f}, mean_steps: {:.1f}'.format(
        args.num_episodes, mean_rewards, mean_steps))


if __name__ == '__main__':
    main()
