# Third party code
#
# The following code are copied or modified from:
# https://github.com/ikostrikov/implicit_q_learning/blob/c1ec002681ff43ef14c3d38dec5881faeca8f624/dataset_utils.py

import numpy as np
from parl.utils import logger
import d4rl
from tqdm import tqdm
__all__ = ['ReplayMemory']


def split_into_trajectories(observations, actions, rewards, masks, dones_float,
                            next_observations):
    trajs = [[]]
    for i in tqdm(range(len(observations))):
        trajs[-1].append((observations[i], actions[i], rewards[i], masks[i],
                          dones_float[i], next_observations[i]))
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])
    return trajs


class ReplayMemory(object):
    def __init__(self, max_size, obs_dim, act_dim):
        """ create a replay memory for off-policy RL or offline RL.

        Args:
            max_size (int): max size of replay memory
            obs_dim (list or tuple): observation shape
            act_dim (list or tuple): action shape
        """
        self.max_size = int(max_size)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obs = np.zeros((max_size, obs_dim), dtype='float32')
        if act_dim == 0:  # Discrete control environment
            self.action = np.zeros((max_size, ), dtype='int32')
        else:  # Continuous control environment
            self.action = np.zeros((max_size, act_dim), dtype='float32')
        self.reward = np.zeros((max_size, ), dtype='float32')
        self.terminal = np.zeros((max_size, ), dtype='bool')
        self.next_obs = np.zeros((max_size, obs_dim), dtype='float32')
        self._curr_size = 0
        self._curr_pos = 0

    def sample_batch(self, batch_size):
        """ sample a batch from replay memory

        Args:
            batch_size (int): batch size

        Returns:
            a batch of experience samples: obs, action, reward, next_obs, terminal
        """
        batch_idx = np.random.randint(self._curr_size, size=batch_size)

        obs = self.obs[batch_idx]
        reward = self.reward[batch_idx]
        action = self.action[batch_idx]
        next_obs = self.next_obs[batch_idx]
        terminal = self.terminal[batch_idx]
        return obs, action, reward, next_obs, terminal

    def size(self):
        """ get current size of replay memory.
        """
        return self._curr_size

    def __len__(self):
        return self._curr_size

    def load_from_d4rl(self, dataset):
        """ load data from d4rl dataset(https://github.com/rail-berkeley/d4rl#using-d4rl) to replay memory.

        Args:
            dataset(dict): dataset that contains:
                            observations (np.float32): shape of (batch_size, obs_dim),
                            next_observations (np.int32): shape of (batch_size, obs_dim),
                            actions (np.float32): shape of (batch_size, act_dim),
                            rewards (np.float32): shape of (batch_size),
                            terminals (bool): shape of (batch_size)

        Example:

        .. code-block:: python

            import gym
            import d4rl

            env = gym.make("hopper-medium-v0")
            rpm = ReplayMemory(max_size=int(2e6), obs_dim=11, act_dim=3)
            rpm.load_from_d4rl(d4rl.qlearning_dataset(env))

            # Output

            # Dataset Info:
            # key: observations,	shape: (999981, 11),	dtype: float32
            # key: actions,	shape: (999981, 3),	dtype: float32
            # key: next_observations,	shape: (999981, 11),	dtype: float32
            # key: rewards,	shape: (999981,),	dtype: float32
            # key: terminals,	shape: (999981,),	dtype: bool
            # Number of terminals on: 3045

        """

        logger.info("Dataset Info: ")
        for key in dataset:
            logger.info('key: {},\tshape: {},\tdtype: {}'.format(
                key, dataset[key].shape, dataset[key].dtype))
        assert 'observations' in dataset
        assert 'next_observations' in dataset
        assert 'actions' in dataset
        assert 'rewards' in dataset
        assert 'terminals' in dataset

        self.obs = dataset['observations']
        self.next_obs = dataset['next_observations']
        self.action = dataset['actions']
        self.reward = dataset['rewards']
        self.terminal = dataset['terminals']
        self._curr_size = dataset['terminals'].shape[0]
        self.dones_float = np.zeros_like(self.reward)
        for i in range(len(self.dones_float) - 1):
            if np.linalg.norm(self.obs[i + 1] - self.next_obs[i]
                              ) > 1e-6 or self.terminal[i] == 1.0:
                self.dones_float[i] = 1
            else:
                self.dones_float[i] = 0
        self.dones_float[-1] = 1
        assert self._curr_size <= self.max_size, 'please set a proper max_size for ReplayMemory'
        logger.info('Number of terminals on: {}'.format(self.terminal.sum()))
        #normalize reward
        trajs = split_into_trajectories(self.obs, self.action, self.reward,
                                        self.terminal, self.dones_float,
                                        self.next_obs)

        def compute_returns(traj):
            episode_return = 0
            for _, _, rew, _, _, _ in traj:
                episode_return += rew
            return episode_return

        trajs.sort(key=compute_returns)
        self.reward /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
        self.reward *= 1000
