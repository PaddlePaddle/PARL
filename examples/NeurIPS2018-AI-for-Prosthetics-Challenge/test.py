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
from env_wrapper import FrameSkip, ActionScale, PelvisBasedObs, ForwardReward
from osim.env import ProstheticsEnv
from parl.utils import logger
from submit_model import SubmitModel


def play_multi_episode(submit_model, episode_num=2, vis=False, seed=0):
    np.random.seed(seed)
    env = ProstheticsEnv(visualize=vis)
    env.change_model(model='3D', difficulty=1, prosthetic=True, seed=seed)
    env = ForwardReward(env)
    env = FrameSkip(env, 4)
    env = ActionScale(env)
    env = PelvisBasedObs(env)
    all_reward = []
    all_shaping_reward = 0
    last_frames_count = 0

    for e in range(episode_num):
        t = time.time()
        episode_reward = 0.0
        episode_shaping_reward = 0.0
        observation = env.reset(project=False)
        target_change_times = 0
        step = 0
        loss = []
        while True:
            step += 1
            action = submit_model.pred_batch(observation, target_change_times)
            observation, reward, done, info = env.step(action, project=False)
            step_frames = info['frame_count'] - last_frames_count
            last_frames_count = info['frame_count']
            episode_reward += reward
            # we pacle it here to drop the first step after changing
            if target_change_times >= 1:
                loss.append(10 * step_frames - reward)
            if info['target_changed']:
                target_change_times = min(target_change_times + 1, 3)
            logger.info("[step/{}]reward:{}  info:{}".format(
                step, reward, info))
            episode_shaping_reward += info['shaping_reward']
            if done:
                break
        all_reward.append(episode_reward)
        all_shaping_reward += episode_shaping_reward
        t = time.time() - t
        logger.info(
            "[episode/{}] time: {} episode_reward:{} change_loss:{} after_change_loss:{} mean_reward:{}"
            .format(e, t, episode_reward, np.sum(loss[:15]), np.sum(loss[15:]),
                    np.mean(all_reward)))
    logger.info("Mean reward:{}".format(np.mean(all_reward)))


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
