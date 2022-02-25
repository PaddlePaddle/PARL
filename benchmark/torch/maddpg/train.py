#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import parl
from simple_model import MAModel
from simple_agent import MAAgent
from parl.algorithms import MADDPG
from parl.env.multiagent_simple_env import MAenv
from parl.utils import logger, summary
from parl.utils import ReplayMemory

from actor import Actor

CRITIC_LR = 0.01  # learning rate for the critic model
ACTOR_LR = 0.01  # learning rate of the actor model
GAMMA = 0.95  # reward discount factor
TAU = 0.01  # soft update
BATCH_SIZE = 1024
MAX_EPISODES = 25000  # stop condition:number of episodes
MAX_STEP_PER_EPISODE = 25  # maximum step per episode
STAT_RATE = 1000  # statistical interval of save model or count reward



def train_agent():
    env = MAenv(args.env)
    logger.info('agent num: {}'.format(env.n))
    logger.info('observation_space: {}'.format(env.observation_space))
    logger.info('action_space: {}'.format(env.action_space))
    logger.info('obs_shape_n: {}'.format(env.obs_shape_n))
    logger.info('act_shape_n: {}'.format(env.act_shape_n))

    for i in range(env.n):
        logger.info('agent {} obs_low:{} obs_high:{}'.format(
            i, env.observation_space[i].low, env.observation_space[i].high))
        logger.info('agent {} act_n:{}'.format(i, env.act_shape_n[i]))
        if ('low' in dir(env.action_space[i])):
            logger.info('agent {} act_low:{} act_high:{} act_shape:{}'.format(
                i, env.action_space[i].low, env.action_space[i].high,
                env.action_space[i].shape))
            logger.info('num_discrete_space:{}'.format(
                env.action_space[i].num_discrete_space))

    from gym import spaces
    from multiagent.multi_discrete import MultiDiscrete
    for space in env.action_space:
        assert (isinstance(space, spaces.Discrete)
                or isinstance(space, MultiDiscrete))

    critic_in_dim = sum(env.obs_shape_n) + sum(env.act_shape_n)
    logger.info('critic_in_dim: {}'.format(critic_in_dim))

    agents = []
    for i in range(env.n):
        model = MAModel(env.obs_shape_n[i], env.act_shape_n[i], critic_in_dim)
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
            batch_size=BATCH_SIZE,
            speedup=(not args.restore))
        agents.append(agent)
    total_steps = 0
    total_episodes = 0

    episode_rewards = []  # sum of rewards for all agents
    agent_rewards = [[] for _ in range(env.n)]  # individual agent reward

    if args.restore:
        # restore modle
        for i in range(len(agents)):
            model_file = args.model_dir + '/agent_' + str(i)
            if not os.path.exists(model_file):
                raise Exception(
                    'model file {} does not exits'.format(model_file))
            agents[i].restore(model_file)

    rpms = []
    memory_size = int(1e5)
    for i in range(len(agents)):
        rpm = ReplayMemory(
            max_size=memory_size,
            obs_dim=env.obs_shape_n[i],
            act_dim=env.act_shape_n[i])
        rpms.append(rpm)

    parl.connect(args.master_address)
    remote_actors = [Actor(args) for _ in range(args.actor_num)]

    t_start = time.time()
    logger.info('Starting...')
    while total_episodes <= MAX_EPISODES:
        latest_weights = [agent.get_weights() for agent in agents]
        for remote_actor in remote_actors:
            remote_actor.set_weights(latest_weights)
        result_object_id = [remote_actor.run_episode() for remote_actor in remote_actors]
        for future_object in result_object_id:
            experience, ep_reward, ep_agent_rewards, steps = future_object.get()

            summary.add_scalar('train_reward/episode', ep_reward, total_episodes)
            summary.add_scalar('train_reward/step', ep_reward, total_steps)
            if args.show:
                print('episode {}, reward {}, agents rewards {}, steps {}'.format(
                    total_episodes, ep_reward, ep_agent_rewards, steps))

            # add experience
            for obs_n, action_n, reward_n, next_obs_n, done_n in experience:
                for i, rpm in enumerate(rpms):
                    rpm.append(obs_n[i], action_n[i], reward_n[i],
                                 next_obs_n[i], done_n[i])

            # train/sample rate 
            for _ in range(len(experience)):
                # learn policy
                for i, agent in enumerate(agents):
                    critic_loss = agent.learn(agents, rpms)
                    if critic_loss != 0.0:
                        summary.add_scalar('critic_loss_%d' % i, critic_loss,
                                           agent.global_train_step)

            # Record reward
            total_steps += steps
            total_episodes += 1
            episode_rewards.append(ep_reward)
            for i in range(env.n):
                agent_rewards[i].append(ep_agent_rewards[i])

            # Keep track of final episode reward
            if total_episodes % STAT_RATE == 0:
                mean_episode_reward = round(
                    np.mean(episode_rewards[-STAT_RATE:]), 3)
                final_ep_ag_rewards = []  # agent rewards for training curve
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(round(np.mean(rew[-STAT_RATE:]), 2))
                use_time = round(time.time() - t_start, 3)
                logger.info(
                    'Steps: {}, Episodes: {}, Mean episode reward: {}, mean agents rewards {}, Time: {}'
                    .format(total_steps, total_episodes, mean_episode_reward,
                            final_ep_ag_rewards, use_time))
                t_start = time.time()
                summary.add_scalar('mean_episode_reward/episode',
                                   mean_episode_reward, total_episodes)
                summary.add_scalar('mean_episode_reward/step', mean_episode_reward,
                                   total_steps)
                summary.add_scalar('use_time/1000episode', use_time,
                                   total_episodes)

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
        '--master_address',
        type=str,
        default='localhost:8010')
    parser.add_argument(
        '--actor_num',
        type=int,
        default=1)

    args = parser.parse_args()
    logger.set_dir('./train_log/' + str(args.env))

    train_agent()
