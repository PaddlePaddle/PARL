#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import gym
import numpy as np

import parl
from parl.utils import logger, summary
from parl.utils.rl_utils import calc_gae, calc_discount_sum_rewards, Scaler
from parl.env.compat_wrappers import CompatWrapper
from parl.algorithms import PPO_Mujoco
from mujoco_model import MujocoModel
from mujoco_agent import MujocoAgent
from actor import Actor
from mujoco_config import mujoco_config


def run_evaluate_episodes(env, agent, scaler, eval_episodes):
    eval_episode_rewards = []
    while len(eval_episode_rewards) < eval_episodes:
        obs = env.reset()
        rewards = 0
        step = 0.0
        scale, offset = scaler.get()
        scale[-1] = 1.0  # don't scale time step feature
        offset[-1] = 0.0  # don't offset time step feature
        while True:
            obs = obs.reshape((1, -1))
            obs = np.append(obs, [[step]], axis=1)  # add time step feature
            obs = (obs - offset) * scale  # center and scale observations
            obs = obs.astype('float32')

            action = agent.predict(obs)
            obs, reward, done, _ = env.step(np.squeeze(action))
            rewards += reward
            step += 1e-3  # increment time step feature

            if done:
                break
        eval_episode_rewards.append(rewards)
    return np.mean(eval_episode_rewards)


def get_remote_trajectories(actors, scaler):
    remote_ids = [actor.run_episode(scaler) for actor in actors]
    return_list = [return_.get() for return_ in remote_ids]

    trajectories, all_unscaled_obs = [], []
    for res in return_list:
        obs, actions, rewards, dones, unscaled_obs = res['obs'], res['actions'], res['rewards'], res['dones'], res[
            'unscaled_obs']
        trajectories.append({'obs': obs, 'actions': actions, 'rewards': rewards, 'dones': dones})
        all_unscaled_obs.append(unscaled_obs)
    # update running statistics for scaling observations
    scaler.update(np.concatenate(all_unscaled_obs))
    return trajectories


def build_train_data(config, trajectories, agent):
    train_obs, train_actions, train_advantages, train_discount_sum_rewards = [], [], [], []
    for trajectory in trajectories:
        pred_values = agent.value(trajectory['obs']).squeeze()

        # scale rewards
        scale_rewards = trajectory['rewards'] * (1 - config['gamma'])
        if len(scale_rewards) <= 1:
            continue
        discount_sum_rewards = calc_discount_sum_rewards(scale_rewards, config['gamma']).astype('float32')
        advantages = calc_gae(scale_rewards, pred_values, 0, config['gamma'], config['gae_lambda'])
        advantages = advantages.astype('float32')
        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        train_obs.append(trajectory['obs'])
        train_actions.append(trajectory['actions'])
        train_advantages.append(advantages)
        train_discount_sum_rewards.append(discount_sum_rewards)

    train_obs = np.concatenate(train_obs)
    train_actions = np.concatenate(train_actions)
    train_advantages = np.concatenate(train_advantages)
    train_discount_sum_rewards = np.concatenate(train_discount_sum_rewards)
    return train_obs, train_actions, train_advantages, train_discount_sum_rewards


def main():
    config = mujoco_config
    config['env'] = args.env
    config['seed'] = args.seed
    config['env_num'] = args.env_num
    config['test_every_episodes'] = args.test_every_episodes
    config['train_total_episodes'] = args.train_total_episodes
    config['episodes_per_batch'] = args.episodes_per_batch

    logger.info("------------------- PPO ---------------------")
    logger.info('Env: {}, seed: {}'.format(config['env'], config['seed']))
    logger.info("---------------------------------------------")
    logger.set_dir('./train_logs/{}_{}'.format(config['env'], config['seed']))

    env = gym.make(args.env)
    env = CompatWrapper(env)
    try:
        env.seed(args.seed)
    except:
        pass

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    obs_dim += 1  # add 1 to obs dim for time step feature

    scaler = Scaler(obs_dim)
    model = MujocoModel(obs_dim, act_dim)
    alg = PPO_Mujoco(model, act_dim=act_dim)
    agent = MujocoAgent(alg, config)

    parl.connect(config['xparl_addr'])
    actors = [Actor(config) for _ in range(config["env_num"])]
    # run a few episodes to initialize scaler
    get_remote_trajectories(actors, scaler)

    test_flag = 0
    episode = 0
    while episode < config['train_total_episodes']:
        latest_params = agent.get_weights()
        # setting the actor to the latest_params
        for remote_actor in actors:
            remote_actor.set_weights(latest_params)
        
        trajectories = []
        while len(trajectories) < config['episodes_per_batch']:
            trajectories.extend(get_remote_trajectories(actors, scaler))
        episode += len(trajectories)

        train_obs, train_actions, train_advantages, train_discount_sum_rewards = build_train_data(
            config, trajectories, agent)

        policy_loss, kl, beta, lr_multiplier, entropy = agent.policy_learn(train_obs, train_actions, train_advantages)
        value_loss, exp_var, old_exp_var = agent.value_learn(train_obs, train_discount_sum_rewards)

        total_train_rewards = sum([np.sum(t['rewards']) for t in trajectories])
        logger.info('Training: Episode {}, Avg train reward: {}, Policy loss: {}, KL: {}, Value loss: {}'.format(
            episode, total_train_rewards / len(trajectories), policy_loss, kl, value_loss))
        summary.add_scalar("train/avg_train_reward", total_train_rewards / len(trajectories), episode)

        if episode // config['test_every_episodes'] >= test_flag:
            while episode // config['test_every_episodes'] >= test_flag:
                test_flag += 1

            avg_reward = run_evaluate_episodes(env, agent, scaler, config['eval_episode'])
            summary.add_scalar('eval/episode_reward', avg_reward, episode)
            logger.info('Evaluation over: {} episodes, Reward: {}'.format(config['eval_episode'], avg_reward))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, help='Mujoco environment name', default='Swimmer-v2')
    parser.add_argument(
        "--env_num", type=int, default=5, help="number of the environment, xparl is needed")
    parser.add_argument('--episodes_per_batch', type=int, default=5, help='Number of episodes per training batch')
    parser.add_argument('--train_total_episodes', type=int, default=int(100), help='maximum training steps')
    parser.add_argument(
        '--test_every_episodes',
        type=int,
        default=int(50),
        help='the step interval between two consecutive evaluations')
    parser.add_argument('--seed', type=int, default=1, help='the step interval between two consecutive evaluations')
    args = parser.parse_args()

    main()
