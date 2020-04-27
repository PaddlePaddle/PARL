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

import os
import gym
import numpy as np
from tqdm import tqdm
import parl
from parl.utils import tensorboard

from cartpole_model import CartpoleModel
from cartpole_agent import CartpoleAgent

from replay_memory import ReplayMemory, Experience

LEARN_FREQ = 5  # update parameters every 5 steps
MEMORY_SIZE = 20000  # replay memory size
MEMORY_WARMUP_SIZE = 200  # store some experiences in the replay memory in advance
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
GAMMA = 0.99  # discount factor of reward


def run_episode(agent, env, rpm, train_or_test, render=False):
    assert train_or_test in ['train', 'test'], train_or_test
    total_reward = 0
    state = env.reset()
    step = 0
    while True:
        step += 1
        if render:
            env.render()
        if train_or_test == 'train':
            action = agent.sample(state)
        else:
            action = agent.predict(state)
        next_state, reward, isOver, _ = env.step(action)

        if train_or_test == 'train':
            rpm.append(Experience(state, action, reward, isOver))

        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            (batch_state, batch_action, batch_reward, batch_next_state,
             batch_isOver) = rpm.sample(BATCH_SIZE)
            train_loss = agent.learn(batch_state, batch_action, batch_reward,
                                     batch_next_state, batch_isOver,
                                     LEARNING_RATE)
            tensorboard.add_scalar('loss', train_loss, agent.global_step)

        total_reward += reward
        state = next_state
        if isOver:
            break
    return total_reward


def main():
    env = gym.make('CartPole-v1')
    action_dim = env.action_space.n
    state_shape = env.observation_space.shape

    rpm = ReplayMemory(MEMORY_SIZE, state_shape)

    model = CartpoleModel(act_dim=action_dim)
    algorithm = parl.algorithms.DQN(model, act_dim=action_dim, gamma=GAMMA)
    agent = CartpoleAgent(
        algorithm,
        state_dim=state_shape[0],
        act_dim=action_dim,
        e_greed=0.1,  # explore
        e_greed_decrement=1e-6
    )  # probability of exploring is decreasing during training

    while len(rpm) < MEMORY_WARMUP_SIZE:  # warm up replay memory
        run_episode(agent, env, rpm, train_or_test='train')

    max_episode = 2000

    # start train
    pbar = tqdm(total=max_episode)
    episode = 0
    while episode < max_episode:
        # train part
        for i in range(0, 50):
            total_reward = run_episode(agent, env, rpm, train_or_test='train')
            episode += 1
            pbar.set_description('[train]e_greed:{}'.format(agent.e_greed))
            pbar.update()

        # test part, run 5 episodes and average
        test_reward_list = []
        for i in range(0, 5):
            total_reward = run_episode(agent, env, rpm, train_or_test='test')
            test_reward_list.append(total_reward)
            if i == 4:
                pbar.write('episode:{}    test_reward:{}'.format(
                    episode, np.mean(test_reward_list)))
                tensorboard.add_scalar('reward', np.mean(test_reward_list),
                                       episode)

    pbar.close()


if __name__ == '__main__':
    main()
