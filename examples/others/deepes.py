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
import gym
import numpy as np


class CartpoleAgent(object):
    def __init__(self, obs_dim, act_dim, learning_rate):
        self.learning_rate = learning_rate
        # init weights
        self.w = np.random.random((act_dim, obs_dim)) * 0.1
        self.b = np.zeros(act_dim)
        self.weights_total_size = self.w.size + self.b.size

    def predict(self, obs):
        out = np.dot(self.w, obs) + self.b
        action = np.argmax(out)
        return action

    def learn(self, rewards, noises):
        gradient = np.dot(
            np.asarray(rewards, dtype=np.float32),
            np.asarray(noises, dtype=np.float32))
        gradient /= rewards.size

        flat_weights = self.get_flat_weights()
        # Compute the new weights.
        new_weights = flat_weights + self.learning_rate * gradient
        self.set_flat_weights(new_weights)

    def set_flat_weights(self, flat_weights):
        self.w = flat_weights[:self.w.size].reshape(self.w.shape)
        self.b = flat_weights[self.w.size:]

    def get_flat_weights(self):
        flat_weights = np.concatenate(([self.w.ravel(), self.b]), axis=0)
        return flat_weights


def evaluate(env, agent):
    ep_reward = 0
    obs = env.reset()
    while True:
        action = agent.predict(obs)
        obs, reward, done, _ = env.step(action)
        ep_reward += reward
        if done:
            break
    return ep_reward


def reward_normalize(reward):
    reward = np.asarray(reward)
    max_r = np.max(reward)
    min_r = np.min(reward)
    if max_r == min_r:
        reward = np.zeros(reward.shape)
    else:
        reward = (reward - min_r) / (max_r - min_r)
        reward -= 0.5
    return reward


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    agent = CartpoleAgent(obs_dim=4, act_dim=2, learning_rate=0.1)

    for epcho in range(100):
        rewards = []
        noises = []
        lastest_flat_weights = agent.get_flat_weights()

        for episode in range(10):
            noise = np.random.randn(agent.weights_total_size)
            perturbation = noise * 0.05

            agent.set_flat_weights(lastest_flat_weights + perturbation)
            ep_reward = evaluate(env, agent)

            noises.append(noise)
            rewards.append(ep_reward)

        normalized_rewards = reward_normalize(rewards)
        agent.set_flat_weights(lastest_flat_weights)
        agent.learn(normalized_rewards, noises)
        # evaluate
        if (epcho % 10) == 0:
            ep_reward = evaluate(env, agent)
            print('Epcho {}, Test reward {}'.format(epcho, ep_reward))
