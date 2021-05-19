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

import os
import gym
import numpy as np
import paddle

import parl
from parl.utils import logger, summary
from parl.env.atari_wrappers import wrap_deepmind
from atari_model import AtariModel
from atari_agent import AtariAgent
from replay_memory_old import ReplayMemory, Experience
from atari_config import config
from parl.algorithms import DQN, DDQN
from tqdm import tqdm

from utils import get_player

# train an episode
def run_train_episode(agent, env, rpm):

    total_reward = 0
    obs = env.reset()
    step = 0
    loss_lst = []

    while True:
        step += 1
        context = rpm.recent_obs()
        context.append(obs)
        context = np.stack(context, axis=0)

        action = agent.sample(context)
        next_obs, reward, done, _ = env.step(action)
        rpm.append(Experience(obs, action, reward, done))

        # train model
        if (rpm.size() > config['memory_warmup_size']) and (step % config['update_freq'] == 0):
            # s,a,r,s',done

            (batch_all_obs, batch_action, batch_reward, batch_done) = rpm.sample_batch(config['batch_size'])
            batch_obs = batch_all_obs[:, :4, :, :]
            batch_next_obs = batch_all_obs[:, 1:, :, :]

            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs, batch_done)
            loss_lst.append(train_loss)

        total_reward += reward
        obs = next_obs

        if done:
            break

    return total_reward, step, np.mean(loss_lst)


def run_evaluate_episodes(agent, env, test=False):

    eval_reward = []
    eval_rounds = config['test_episodes'] if test else config['eval_episodes']

    with paddle.no_grad():

        for _ in range(eval_rounds):
            obs = env.reset()
            episode_reward = 0

            while True:
                action = agent.predict(obs)
                obs, reward, done, _ = env.step(action)
                episode_reward += reward

                if config['eval_render']:
                    env.render()

                if done:
                    break

            eval_reward.append(episode_reward)

    return np.mean(eval_reward), eval_reward


def main():

    import datetime

    curr_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    algo_name = config['algorithm']
    dim = config['env_dim']

    # env = gym.make(config['env_name'])
    # env = wrap_deepmind(env, dim=dim, framestack=False, obs_format='NCHW')
    # test_env = gym.make(config['env_name'])
    # test_env = wrap_deepmind(test_env, dim=dim, obs_format='NCHW')

    env = get_player(config['rom_path'], image_size=(dim, dim), train=True, frame_skip=4)
    test_env = get_player(
        config['rom_path'],
        image_size=(dim, dim),
        frame_skip=4,
        context_len=4)

    env.seed(config['train_env_seed'])
    test_env.seed(config['test_env_seed'])

    act_dim = env.action_space.n
    config['act_dim'] = act_dim

    logger.set_dir(f'./train_log/{algo_name}/{curr_time}')
    logger.info('env {}, train obs_dim {}, train act_dim {}'.format(config['env_name'], env.observation_space.shape, act_dim))
    logger.info(f'current configs are : \n {config}')

    rpm = ReplayMemory(config['memory_size'], (dim, dim), 4)

    # build an agent
    model = AtariModel(act_dim=act_dim, dueling=config['dueling'])

    if algo_name == 'DQN':
        alg = DQN(model, gamma=config['gamma'], lr=config['lr_start'])

    elif algo_name == 'DDQN':
        alg = DDQN(model, gamma=config['gamma'], lr=config['lr_start'])

    else:
        pass

    agent = AtariAgent(alg, config)

    with tqdm(
            total=config['memory_warmup_size'], desc='[Replay Memory Warm Up]') as pbar:
        while rpm.size() < config['memory_warmup_size']:
            total_reward, steps, _ = run_train_episode(agent, env, rpm)
            pbar.update(steps)

    test_flag = 0
    train_total_steps = config['train_total_steps']
    pbar = tqdm(total=train_total_steps)
    cum_steps = 0

    while cum_steps < train_total_steps:
        
        # start epoch
        total_reward, steps, loss = run_train_episode(agent, env, rpm)
        cum_steps += steps
        pbar.set_description('[train]exploration:{}, learning_rate {}'.format(agent.curr_ep, alg.optimizer.get_lr()))
        summary.add_scalar(f'{algo_name}/training_rewards', total_reward, cum_steps)
        summary.add_scalar(f'{algo_name}/loss', loss, cum_steps)  # mean of total loss
        summary.add_scalar(f'{algo_name}/exploration', agent.curr_ep, cum_steps)
        summary.add_scalar(f'{algo_name}/learning_rate', alg.optimizer.get_lr(), cum_steps)

        pbar.update(steps)

        if cum_steps // config['eval_every_steps'] >= test_flag:

            while cum_steps // config['eval_every_steps'] >= test_flag:
                test_flag += 1

            pbar.write("testing")

            eval_rewards_mean, _ = run_evaluate_episodes(agent, test_env)

            logger.info(
                "eval_agent done, (steps, eval_reward): ({}, {})".format(
                    cum_steps, eval_rewards_mean))

            summary.add_scalar(f"{algo_name}/mean_{config['eval_episodes']}_validation_rewards", eval_rewards_mean, cum_steps)

    pbar.close()

    # final test score
    eval_rewards_mean, eval_rewards = run_evaluate_episodes(agent, test_env, test=True)
    std = np.std(eval_rewards)
    logger.info(f"final mean {config['test_episodes']} test rewards is {eval_rewards_mean} +- {std}")

    # save the parameters to ./model.ckpt
    save_path = f'./model/{algo_name}/{curr_time}_model.ckpt'
    agent.save(save_path)


if __name__ == '__main__':
    main()
