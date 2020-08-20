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

from smac.env import StarCraft2Env
import numpy as np
import os
from sc2_model import ComaModel
from sc2_agent import Agents
from parl.algorithms import COMA
from parl.utils import tensorboard


def run_episode(env, agents, config, test=False):
    o, u, r, s, avail_u, u_onehot, isover, padded = [], [], [], [], [], [], [], []
    env.reset()
    done = False
    step = 0
    ep_reward = 0
    last_act = [0 for _ in range(config['n_agents'])]
    agents.init_hidden()  # init rnn h0 for all agents

    while not done:
        obs = env.get_obs()
        state = env.get_state()
        acts, avail_acts, acts_onehot = [], [], []

        for agent_id in range(config['n_agents']):
            avail_act = env.get_avail_agent_actions(agent_id)

            # action
            epsilon = 0 if test else config['epsilon']
            act = agents.sample(obs[agent_id], last_act[agent_id], agent_id,
                                avail_act, epsilon, test)
            last_act[agent_id] = act

            # action one-hot
            act_onehot = np.zeros(config['n_actions'])
            act_onehot[act] = 1
            acts.append(act)
            acts_onehot.append(act_onehot)
            avail_acts.append(avail_act)

        # step
        reward, done, _ = env.step(acts)

        if step == config['episode_limit'] - 1:
            done = 1

        o.append(obs)
        s.append(state)
        u.append(np.reshape(acts, [config['n_agents'], 1]))
        u_onehot.append(acts_onehot)
        avail_u.append(avail_acts)
        r.append([reward])
        isover.append([done])
        padded.append([0.])  # 0: no padded, 1: padded

        ep_reward += reward
        step += 1

    # fill trainsition len to episode_limit
    for _ in range(step, config['episode_limit']):
        # shape: (config['episode_limit'], n_agents, shape)
        o.append(np.zeros((config['n_agents'], config['obs_shape'])))
        s.append(np.zeros(config['state_shape']))
        u.append(np.zeros([config['n_agents'], 1]))
        u_onehot.append(np.zeros((config['n_agents'], config['n_actions'])))
        avail_u.append(np.zeros((config['n_agents'], config['n_actions'])))
        # shape: (config['episode_limit'], 1)
        r.append([0.])
        padded.append([1.])
        isover.append([1.])

    ep_data = dict(
        o=o.copy(),
        s=s.copy(),
        u=u.copy(),
        r=r.copy(),
        avail_u=avail_u.copy(),
        u_onehot=u_onehot.copy(),
        padded=padded.copy(),
        isover=isover.copy())

    # add an additional dimension at axis 0 for each item
    for key in ep_data.keys():
        # each items shape: (1, trainsition_num, n_agents, own_shape)
        ep_data[key] = np.array([ep_data[key]])

    return ep_data, ep_reward


def run(env, agents, config):
    win_rates = []
    episode_rewards = []
    train_steps = 0
    for epoch in range(config['n_epoch']):
        print('train epoch {}'.format(epoch))
        # decay epsilon at the begging of each epoch
        if config['epsilon'] > config['min_epsilon']:
            config['epsilon'] -= config['anneal_epsilon']

        # run n episode(s)
        ep_data_list = []
        for _ in range(config['n_episodes']):
            ep_data, _ = run_episode(env, agents, config, test=False)
            ep_data_list.append(ep_data)
        # each item in ep_batch shape: (episode_num, trainsition_num, n_agents, item_shape)
        ep_batch = ep_data_list[0]
        ep_data_list.pop(0)
        for ep_data in ep_data_list:
            for key in ep_batch.keys():
                ep_batch[key] = np.concatenate((ep_batch[key], ep_data[key]),
                                               axis=0)

        # learn
        agents.learn(ep_batch, config['epsilon'])
        train_steps += 1

        # save model
        if train_steps > 0 and train_steps % config['save_cycle'] == 0:
            model_path = config['model_dir'] + '/coma_' + str(
                train_steps) + '.ckpt'
            agents.save(save_path=model_path)
            print('save model: ', model_path)

        # test
        if epoch % config['test_cycle'] == 0:
            win_rate, ep_mean_reward = test(env, agents, config)
            # print('win_rate is ', win_rate)
            win_rates.append(win_rate)
            episode_rewards.append(ep_mean_reward)
            tensorboard.add_scalar('win_rate', win_rates[-1], len(win_rates))
            tensorboard.add_scalar('episode_rewards', episode_rewards[-1],
                                   len(episode_rewards))
            print('win_rate', win_rates, len(win_rates))
            print('episode_rewards', episode_rewards, len(episode_rewards))


def test(env, agents, config):
    win_number = 0
    episode_rewards = 0
    for ep_id in range(config['test_episode_n']):
        _, ep_reward = run_episode(env, agents, config, test=True)
        episode_rewards += ep_reward
        if ep_reward > config['threshold']:
            win_number += 1
    return win_number / config['test_episode_n'], episode_rewards / config[
        'test_episode_n']


def test_by_sparse_reward(agents, config):
    env = StarCraft2Env(
        map_name=config['map'],
        difficulty=config['difficulty'],
        seed=config['env_seed'],
        replay_dir=config['replay_dir'],
        reward_sparse=True,  # Receive 1/-1 reward for winning/loosing an episode
        reward_scale=False)
    win_number = 0
    for ep_id in range(config['test_episode_n']):
        _, ep_reward = run_episode(env, agents, config, test=True)
        result = 'win' if ep_reward > 0 else 'defeat'
        print('Episode {}: {}'.format(ep_id, result))
        if ep_reward > 0:
            win_number += 1
    env.close()
    win_rate = win_number / config['test_episode_n']
    print('The win rate of coma is  {}'.format(win_rate))
    return win_rate


def main(config):
    env = StarCraft2Env(
        map_name=config['map'],
        seed=config['env_seed'],
        difficulty=config['difficulty'],
        replay_dir=config['replay_dir'])
    env_info = env.get_env_info()

    config['n_actions'] = env_info['n_actions']
    config['n_agents'] = env_info['n_agents']
    config['state_shape'] = env_info['state_shape']
    config['obs_shape'] = env_info['obs_shape']
    config['episode_limit'] = env_info['episode_limit']

    model = ComaModel(config=config)
    algorithm = COMA(
        model,
        n_actions=config['n_actions'],
        n_agents=config['n_agents'],
        grad_norm_clip=config['grad_norm_clip'],
        actor_lr=config['actor_lr'],
        critic_lr=config['critic_lr'],
        gamma=config['gamma'],
        td_lambda=config['td_lambda'])
    agents = Agents(algorithm, config)

    # restore model here
    model_file = config['model_dir'] + '/coma.ckpt'
    if config['restore'] and os.path.isfile(model_file):
        agents.restore(model_file)
        print('model loaded: ', model_file)

    if config['test']:
        test_by_sparse_reward(agents, config)
    else:
        run(env, agents, config)

    env.close()


if __name__ == '__main__':
    from coma_config import config
    main(config)
