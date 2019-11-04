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

import argparse
import numpy as np
import time
from env_wrapper import FrameSkip, ActionScale, OfficialObs, ForwardReward
from osim.env import L2M2019Env
from parl.utils import logger
from submit_model import SubmitModel


def play_multi_episode(submit_model, episode_num=2, vis=False, seed=0):
    np.random.seed(seed)

    env = L2M2019Env(difficulty=3, visualize=vis)
    env.change_model(model='3D', difficulty=3)
    env = ForwardReward(env)
    env = FrameSkip(env, 4)
    env = ActionScale(env)
    env = OfficialObs(env)
    all_reward = []

    for e in range(episode_num):
        episode_reward = 0.0
        observation = env.reset(project=True, obs_as_dict=True)
        step = 0
        target_change_times = 0
        while True:
            step += 1
            action = submit_model.pred_batch(observation, target_change_times)
            observation, reward, done, info = env.step(
                action, project=True, obs_as_dict=True)
            if info['target_changed']:
                target_change_times += 1
            episode_reward += reward
            if done:
                break
        all_reward.append(episode_reward)
        logger.info("[episode/{}] episode_reward:{} mean_reward:{}".format(\
                      e, episode_reward, np.mean(all_reward)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--use_cuda', action="store_true", help='If set, will run in gpu 0')
    parser.add_argument(
        '--vis', action="store_true", help='If set, will visualize.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument(
        '--episode_num', type=int, default=1, help='Episode number to run.')
    args = parser.parse_args()

    submit_model = SubmitModel(use_cuda=args.use_cuda)

    play_multi_episode(
        submit_model,
        episode_num=args.episode_num,
        vis=args.vis,
        seed=args.seed)
