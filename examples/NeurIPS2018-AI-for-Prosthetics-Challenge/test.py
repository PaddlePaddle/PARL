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
from env_wrapper import FrameSkip, ActionScale, PelvisBasedObs, TestReward
from multi_head_ddpg import MultiHeadDDPG
from opensim_agent import OpenSimAgent
from opensim_model import OpenSimModel
from osim.env import ProstheticsEnv
from parl.utils import logger
"""
Test model with ensemble predict
"""


def play_multi_episode(agent, episode_num=2, vis=False, seed=0):
    np.random.seed(seed)
    env = ProstheticsEnv(visualize=vis)
    env.change_model(model='3D', difficulty=1, prosthetic=True, seed=seed)
    env = TestReward(env)
    env = FrameSkip(env, 4)
    env = ActionScale(env)
    env = PelvisBasedObs(env)

    all_reward = []

    for e in range(episode_num):
        t = time.time()
        episode_reward = 0.0
        obs = env.reset(project=False)
        step = 0
        while True:
            step += 1

            batch_obs = np.expand_dims(obs, axis=0)

            action = agent.ensemble_predict(batch_obs)
            action = np.squeeze(action, axis=0)
            obs, reward, done, info = env.step(action, project=False)
            episode_reward += reward
            logger.info("[step/{}]reward:{}".format(step, reward))
            if done:
                break
        all_reward.append(episode_reward)
        t = time.time() - t
        logger.info(
            "[episode/{}] time: {} episode_reward:{} mean_reward:{}".format(
                e, t, episode_reward, np.mean(all_reward)))
    logger.info("Mean reward:{}".format(np.mean(all_reward)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--restore_model_path', type=str, help='restore model path for test')
    parser.add_argument(
        '--vis', action="store_true", help='If set, will visualize.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument(
        '--episode_num', type=int, default=1, help='Episode number to run.')
    parser.add_argument('--ensemble_num', type=int, help='ensemble_num')
    args = parser.parse_args()

    ACT_DIM = 19
    VEL_DIM = 4
    OBS_DIM = 185 + VEL_DIM
    GAMMA = 0.96
    TAU = 0.001
    models = []
    for i in range(args.ensemble_num):
        models.append(OpenSimModel(OBS_DIM, VEL_DIM, ACT_DIM, model_id=i))
    hyperparas = {
        'gamma': GAMMA,
        'tau': TAU,
        'ensemble_num': args.ensemble_num
    }
    alg = MultiHeadDDPG(models, hyperparas)
    agent = OpenSimAgent(alg, OBS_DIM, ACT_DIM, args.ensemble_num)

    agent.load_params(args.restore_model_path)

    play_multi_episode(
        agent, episode_num=args.episode_num, vis=args.vis, seed=args.seed)
