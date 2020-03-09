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

import numpy as np
import parl
from parl.utils import logger
from env_wrapper import ObsProcessWrapper, ActionProcessWrapper, RewardWrapper
from rlschool import LiftSim
from lift_model import LiftModel
from lift_agent import LiftAgent
from a2c_config import config


def evaluate_one_day(model_path):
    env = LiftSim()
    env = ActionProcessWrapper(env)
    env = ObsProcessWrapper(env)
    act_dim = env.act_dim
    obs_dim = env.obs_dim
    config['obs_dim'] = obs_dim

    model = LiftModel(act_dim)
    algorithm = parl.algorithms.A3C(
        model, vf_loss_coeff=config['vf_loss_coeff'])
    agent = LiftAgent(algorithm, config)
    agent.restore(model_path)

    reward_24h = 0
    obs = env.reset()
    for i in range(24 * 3600 * 2):  # 24h, 1step = 0.5s
        action, _ = agent.sample(obs)
        #print(action)
        obs, reward, done, info = env.step(action)
        reward_24h += reward
        if (i + 1) % (3600 * 2) == 0:
            logger.info('hour {}, total_reward: {}'.format(
                (i + 1) // (3600 * 2), reward_24h))

    logger.info('model_path: {}, 24h reward: {}'.format(
        model_path, reward_24h))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path', type=str, help='path of the model to restore')
    args = parser.parse_args()

    evaluate_one_day(args.model_path)
