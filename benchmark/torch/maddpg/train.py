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

import os
import time
import argparse
import numpy as np
from simple_model import MAModel
from simple_agent import MAAgent
from parl.algorithms import MADDPG
from parl.env.multiagent_env import MAenv
from parl.utils import logger, summary
from gym import spaces

CRITIC_LR = 0.01  # learning rate for the critic model
ACTOR_LR = 0.01  # learning rate of the actor model
GAMMA = 0.95  # reward discount factor
TAU = 0.01  # soft update
BATCH_SIZE = 1024
MAX_STEP_PER_EPISODE = 25  # maximum step per episode
EVAL_EPISODES = 3


# Runs policy and returns episodes' rewards and steps for evaluation
def run_evaluate_episodes(env, agents, eval_episodes):
    eval_episode_rewards = []
    eval_episode_steps = []
    while len(eval_episode_rewards) < eval_episodes:
        obs_n = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done and steps < MAX_STEP_PER_EPISODE:
            steps += 1
            action_n = [
                agent.predict(obs) for agent, obs in zip(agents, obs_n)
            ]
            obs_n, reward_n, done_n, _ = env.step(action_n)
            done = all(done_n)
            total_reward += sum(reward_n)
            # show animation
            if args.show:
                time.sleep(0.1)
                env.render()

        eval_episode_rewards.append(total_reward)
        eval_episode_steps.append(steps)
    return eval_episode_rewards, eval_episode_steps


def run_episode(env, agents):
    obs_n = env.reset()
    done = False
    total_reward = 0
    agents_reward = [0 for _ in range(env.n)]
    steps = 0
    while not done and steps < MAX_STEP_PER_EPISODE:
        steps += 1
        action_n = [agent.sample(obs) for agent, obs in zip(agents, obs_n)]
        next_obs_n, reward_n, done_n, _ = env.step(action_n)
        done = all(done_n)

        # store experience
        for i, agent in enumerate(agents):
            agent.add_experience(obs_n[i], action_n[i], reward_n[i],
                                 next_obs_n[i], done_n[i])

        # compute reward of every agent
        obs_n = next_obs_n
        for i, reward in enumerate(reward_n):
            total_reward += reward
            agents_reward[i] += reward

        # show model effect without training
        if args.restore and args.show:
            continue

        # learn policy
        for i, agent in enumerate(agents):
            critic_loss = agent.learn(agents)

    return total_reward, agents_reward, steps


def main():
    logger.set_dir('./train_log/{}_{}'.format(args.env,
                                              args.continuous_actions))

    env = MAenv(args.env, args.continuous_actions)
    if args.continuous_actions:
        assert isinstance(env.action_space[0], spaces.Box)
    critic_in_dim = sum(env.obs_shape_n) + sum(env.act_shape_n)

    # build agents
    agents = []
    for i in range(env.n):
        model = MAModel(env.obs_shape_n[i], env.act_shape_n[i], critic_in_dim,
                        args.continuous_actions)
        algorithm = MADDPG(
            model,
            agent_index=i,
            act_space=env.action_space,
            gamma=GAMMA,
            tau=TAU,
            critic_lr=CRITIC_LR,
            actor_lr=ACTOR_LR)
        agent = MAAgent(
            algorithm,
            agent_index=i,
            obs_dim_n=env.obs_shape_n,
            act_dim_n=env.act_shape_n,
            batch_size=BATCH_SIZE)
        agents.append(agent)

    if args.restore:
        # restore modle
        for i in range(len(agents)):
            model_file = args.model_dir + '/agent_' + str(i)
            if not os.path.exists(model_file):
                raise Exception(
                    'model file {} does not exits'.format(model_file))
            agents[i].restore(model_file)

    total_steps = 0
    total_episodes = 0
    while total_episodes <= args.max_episodes:
        # run an episode
        ep_reward, ep_agent_rewards, steps = run_episode(env, agents)
        summary.add_scalar('train/episode_reward_wrt_episode', ep_reward,
                           total_episodes)
        summary.add_scalar('train/episode_reward_wrt_step', ep_reward,
                           total_steps)
        logger.info(
            'total_steps {}, episode {}, reward {}, agents rewards {}, episode steps {}'
            .format(total_steps, total_episodes, ep_reward, ep_agent_rewards,
                    steps))

        total_steps += steps
        total_episodes += 1

        # evaluate agents
        if total_episodes % args.test_every_episodes == 0:
            eval_episode_rewards, eval_episode_steps = run_evaluate_episodes(
                env, agents, EVAL_EPISODES)
            summary.add_scalar('eval/episode_reward',
                               np.mean(eval_episode_rewards), total_episodes)
            logger.info('Evaluation over: {} episodes, Reward: {}'.format(
                EVAL_EPISODES, np.mean(eval_episode_rewards)))

            # save model
            if not args.restore:
                model_dir = args.model_dir
                os.makedirs(os.path.dirname(model_dir), exist_ok=True)
                for i in range(len(agents)):
                    model_name = '/agent_' + str(i)
                    agents[i].save(model_dir + model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Environment
    parser.add_argument(
        '--env',
        type=str,
        default='simple_speaker_listener',
        help='scenario of MultiAgentEnv')
    # auto save model, optional restore model
    parser.add_argument(
        '--show', action='store_true', default=False, help='display or not')
    parser.add_argument(
        '--restore',
        action='store_true',
        default=False,
        help='restore or not, must have model_dir')
    parser.add_argument(
        '--model_dir',
        type=str,
        default='./model',
        help='directory for saving model')
    parser.add_argument(
        '--continuous_actions',
        action='store_true',
        default=False,
        help='use continuous action mode or not')
    parser.add_argument(
        '--max_episodes',
        type=int,
        default=25000,
        help='the maximum number of episodes')
    parser.add_argument(
        '--test_every_episodes',
        type=int,
        default=int(1e3),
        help='the episode interval between two consecutive evaluations')
    args = parser.parse_args()

    main()
