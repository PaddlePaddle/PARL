import gym
import numpy as np
import paddle.fluid as fluid
from parl.utils import logger

from cartpole_model import CartpoleModel
from cartpole_agent import CartpoleAgent
from policy_gradient import PolicyGradient
from utils import calc_discount_norm_reward

OBS_DIM = 4
ACT_DIM = 2
GAMMA = 0.99
LEARNING_RATE = 1e-3


def run_episode(env, agent, train_or_test='train'):
    obs_list, action_list, reward_list = [], [], []
    obs = env.reset()
    while True:
        obs_list.append(obs)
        if train_or_test == 'train':
            action = agent.sample(obs)
        else:
            action = agent.predict(obs)
        action_list.append(action)

        obs, reward, done, _ = env.step(action)
        reward_list.append(reward)

        if done:
            break
    return obs_list, action_list, reward_list


def main():
    env = gym.make('CartPole-v0')
    model = CartpoleModel(name_scope='noIdeaWhyNeedThis', act_dim=ACT_DIM)
    alg = PolicyGradient(model, LEARNING_RATE)
    agent = CartpoleAgent(alg, OBS_DIM, ACT_DIM)

    with fluid.dygraph.guard():
        for i in range(1000):  # 100 episodes
            obs_list, action_list, reward_list = run_episode(env, agent)
            if i % 10 == 0:
                logger.info("Episode {}, Reward Sum {}.".format(
                    i, sum(reward_list)))

            batch_obs = np.array(obs_list)
            batch_action = np.array(action_list)
            batch_reward = calc_discount_norm_reward(reward_list, GAMMA)

            agent.learn(batch_obs, batch_action, batch_reward)
            if (i + 1) % 100 == 0:
                _, _, reward_list = run_episode(env,
                                                agent,
                                                train_or_test='test')
                total_reward = np.sum(reward_list)
                logger.info('Test reward: {}'.format(total_reward))


if __name__ == '__main__':
    main()
