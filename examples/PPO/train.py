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

from parl.utils import check_version_for_fluid  # requires parl >= 1.4.1
check_version_for_fluid()

import argparse
import gym
import numpy as np
import parl
from mujoco_agent import MujocoAgent
from mujoco_model import MujocoModel
from parl.utils import logger
from parl.utils.rl_utils import calc_gae, calc_discount_sum_rewards
from parl.env.continuous_wrappers import ActionMappingWrapper
from scaler import Scaler


def run_train_episode(env, agent, scaler):
    obs = env.reset()
    observes, actions, rewards, unscaled_obs = [], [], [], []
    step = 0.0
    scale, offset = scaler.get()
    scale[-1] = 1.0  # don't scale time step feature
    offset[-1] = 0.0  # don't offset time step feature
    while True:
        obs = obs.reshape((1, -1))
        obs = np.append(obs, [[step]], axis=1)  # add time step feature
        unscaled_obs.append(obs)
        obs = (obs - offset) * scale  # center and scale observations
        obs = obs.astype('float32')
        observes.append(obs)

        action = agent.policy_sample(obs)
        action = np.clip(action, -1.0, 1.0)

        action = action.reshape((1, -1)).astype('float32')
        actions.append(action)

        obs, reward, done, _ = env.step(np.squeeze(action))
        rewards.append(reward)
        step += 1e-3  # increment time step feature

        if done:
            break

    return (np.concatenate(observes), np.concatenate(actions),
            np.array(rewards, dtype='float32'), np.concatenate(unscaled_obs))


def run_evaluate_episode(env, agent, scaler):
    obs = env.reset()
    rewards = []
    step = 0.0
    scale, offset = scaler.get()
    scale[-1] = 1.0  # don't scale time step feature
    offset[-1] = 0.0  # don't offset time step feature
    while True:
        obs = obs.reshape((1, -1))
        obs = np.append(obs, [[step]], axis=1)  # add time step feature
        obs = (obs - offset) * scale  # center and scale observations
        obs = obs.astype('float32')

        action = agent.policy_predict(obs)

        obs, reward, done, _ = env.step(np.squeeze(action))
        rewards.append(reward)

        step += 1e-3  # increment time step feature

        if done:
            break
    return np.sum(rewards)


def collect_trajectories(env, agent, scaler, episodes):
    trajectories, all_unscaled_obs = [], []
    for e in range(episodes):
        obs, actions, rewards, unscaled_obs = run_train_episode(
            env, agent, scaler)
        trajectories.append({
            'obs': obs,
            'actions': actions,
            'rewards': rewards,
        })
        all_unscaled_obs.append(unscaled_obs)
    # update running statistics for scaling observations
    scaler.update(np.concatenate(all_unscaled_obs))
    return trajectories


def build_train_data(trajectories, agent):
    train_obs, train_actions, train_advantages, train_discount_sum_rewards = [], [], [], []
    for trajectory in trajectories:
        pred_values = agent.value_predict(trajectory['obs'])

        # scale rewards
        scale_rewards = trajectory['rewards'] * (1 - args.gamma)

        discount_sum_rewards = calc_discount_sum_rewards(
            scale_rewards, args.gamma).astype('float32')

        advantages = calc_gae(scale_rewards, pred_values, 0, args.gamma,
                              args.lam)

        # normalize advantages
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-6)
        advantages = advantages.astype('float32')

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
    env = gym.make(args.env)
    env = ActionMappingWrapper(env)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    obs_dim += 1  # add 1 to obs dim for time step feature

    scaler = Scaler(obs_dim)

    model = MujocoModel(obs_dim, act_dim)
    alg = parl.algorithms.PPO(
        model,
        act_dim=act_dim,
        policy_lr=model.policy_lr,
        value_lr=model.value_lr)
    agent = MujocoAgent(
        alg, obs_dim, act_dim, args.kl_targ, loss_type=args.loss_type)

    # run a few episodes to initialize scaler
    collect_trajectories(env, agent, scaler, episodes=5)

    test_flag = 0
    total_steps = 0
    while total_steps < args.train_total_steps:
        trajectories = collect_trajectories(
            env, agent, scaler, episodes=args.episodes_per_batch)
        total_steps += sum([t['obs'].shape[0] for t in trajectories])
        total_train_rewards = sum([np.sum(t['rewards']) for t in trajectories])

        train_obs, train_actions, train_advantages, train_discount_sum_rewards = build_train_data(
            trajectories, agent)

        policy_loss, kl = agent.policy_learn(train_obs, train_actions,
                                             train_advantages)
        value_loss = agent.value_learn(train_obs, train_discount_sum_rewards)

        logger.info(
            'Steps {}, Train reward: {}, Policy loss: {}, KL: {}, Value loss: {}'
            .format(total_steps, total_train_rewards / args.episodes_per_batch,
                    policy_loss, kl, value_loss))
        if total_steps // args.test_every_steps >= test_flag:
            while total_steps // args.test_every_steps >= test_flag:
                test_flag += 1
            eval_reward = run_evaluate_episode(env, agent, scaler)
            logger.info('Steps {}, Evaluate reward: {}'.format(
                total_steps, eval_reward))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env',
        type=str,
        help='Mujoco environment name',
        default='HalfCheetah-v2')
    parser.add_argument(
        '--gamma', type=float, help='Discount factor', default=0.995)
    parser.add_argument(
        '--lam',
        type=float,
        help='Lambda for Generalized Advantage Estimation',
        default=0.98)
    parser.add_argument(
        '--kl_targ', type=float, help='D_KL target value', default=0.003)
    parser.add_argument(
        '--episodes_per_batch',
        type=int,
        help='Number of episodes per training batch',
        default=5)
    parser.add_argument(
        '--loss_type',
        type=str,
        help="Choose loss type of PPO algorithm, 'CLIP' or 'KLPEN'",
        default='CLIP')
    parser.add_argument(
        '--train_total_steps',
        type=int,
        default=int(1e7),
        help='maximum training steps')
    parser.add_argument(
        '--test_every_steps',
        type=int,
        default=int(1e4),
        help='the step interval between two consecutive evaluations')

    args = parser.parse_args()

    main()
