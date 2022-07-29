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

import argparse
import numpy as np
from parl.utils import logger, tensorboard

from configs import Config
from env_utils import ParallelEnv, LocalEnv
from storage import RolloutStorage
from model import PPOModel
from parl.algorithms import PPO
from agent import PPOAgent


# Runs policy until 'real done' and returns episode reward
# A fixed seed is used for the eval environment
def run_evaluate_episodes(agent, eval_env_seed=120):
    env = LocalEnv(args.env, env_seed=eval_env_seed)
    eval_reward = 0.
    while True:
        obs = env.reset()
        done = False
        while not done:
            action = agent.predict(obs)
            obs, reward, done, info = env.step(action)
        if "episode" in info.keys():
            eval_reward = info["episode"]["r"]
            break
    return eval_reward


def main():
    logger.info("------------------- PPO ---------------------")
    logger.info('Env: {}, env_num: {}, seed: {}'.format(
        args.env, args.env_num, args.seed))
    logger.info("---------------------------------------------")

    logger.set_dir('./train_logs/{}_{}'.format(args.env, args.seed))

    config = Config['mujoco'] if args.continuous_action else Config['atari']

    envs = ParallelEnv(
        args.env, args.seed, config=config, xparl_addr=args.xparl_addr)
    obs_space = envs.obs_space
    act_space = envs.act_space

    model = PPOModel(obs_space, act_space)
    ppo = PPO(model, config)
    agent = PPOAgent(ppo)

    rollout = RolloutStorage(config['step_nums'], config['env_num'], obs_space,
                             act_space)

    obs = envs.reset()
    done = np.zeros(config['env_num'], dtype='float32')

    total_steps = 0
    test_flag = 0
    num_updates = int(config['train_total_steps'] // config['batch_size'])
    for update in range(1, num_updates + 1):
        for step in range(0, config['step_nums']):
            total_steps += 1 * config['env_num']

            value, action, logprob, _ = agent.sample(obs)
            next_obs, reward, next_done, info = envs.step(action)
            rollout.append(obs, action, logprob, reward, done, value.flatten())
            obs, done = next_obs, next_done

            for item in info:
                if "episode" in item.keys():
                    logger.info(
                        "Training: total steps={}, episodic_return={item['episode']['r']}"
                        .format(total_steps))
                    tensorboard.add_scalar("train/episode_reward",
                                           item["episode"]["r"], total_steps)
                    break

        # Bootstrap value if not done
        value = agent.value(obs)
        rollout.compute_returns(value, done, config['gamma'],
                                config['gae_lambda'])

        # Optimizing the policy and value network
        v_loss, pg_loss, entropy_loss, lr = agent.learn(rollout)

        if (total_steps + 1) // args.test_every_steps >= test_flag:
            while (total_steps + 1) // args.test_every_steps >= test_flag:
                test_flag += 1
                avg_reward = run_evaluate_episodes(agent)
                tensorboard.add_scalar('eval/episode_reward', avg_reward,
                                       total_steps)
                logger.info('Evaluation over: {} episodes, Reward: {}'.format(
                    3, avg_reward))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="HalfCheetah-v2",
        help="the id of the environment")
    parser.add_argument(
        "--seed", type=int, default=110, help="seed of the experiment")
    parser.add_argument(
        "--env_num",
        type=int,
        default=1,
        help=
        "number of the environment. Note: if greater than 1, xparl is needed")
    parser.add_argument(
        "--continuous_action",
        type=bool,
        default=True,
        help="the type of the environment")
    parser.add_argument(
        "--xparl_addr",
        type=str,
        default=None,
        help="the id of the environment")
    parser.add_argument(
        '--test_every_steps',
        type=int,
        default=int(5e3),
        help='The step interval between two consecutive evaluations')

    args = parser.parse_args()
    main()
