# Third Party Code
# The following code is copied or modified from:
# https://github.com/kzl/decision-transformer/tree/master/gym/decision_transformer

import numpy as np
import pickle
import random
import torch
from parl.utils import logger


class DataLoader(object):
    def __init__(self, dataset_path, pct_traj, max_ep_len, scale):
        self.dataset_path = dataset_path
        self.pct_traj = pct_traj
        self.scale = scale
        self.device = torch.device('cuda' if torch.cuda.
                                   is_available() else 'cpu')
        self.max_ep_len = max_ep_len
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)

        states, traj_lens, returns = [], [], []
        for path in trajectories:
            states.append(path['observations'])
            traj_lens.append(len(path['observations']))
            returns.append(path['rewards'].sum())
        traj_lens, returns = np.array(traj_lens), np.array(returns)

        # used for input normalization
        states = np.concatenate(states, axis=0)
        self.state_mean, self.state_std = np.mean(
            states, axis=0), np.std(
                states, axis=0) + 1e-6

        num_timesteps = sum(traj_lens)

        # only train on top pct_traj trajectories (for %BC experiment)
        num_timesteps = max(int(pct_traj * num_timesteps), 1)
        sorted_inds = np.argsort(returns)  # lowest to highest
        num_trajectories = 1
        timesteps = traj_lens[sorted_inds[-1]]
        ind = len(trajectories) - 2
        while ind >= 0 and timesteps + traj_lens[
                sorted_inds[ind]] <= num_timesteps:
            timesteps += traj_lens[sorted_inds[ind]]
            num_trajectories += 1
            ind -= 1
        sorted_inds = sorted_inds[-num_trajectories:]

        # used to reweight sampling so we sample according to timesteps instead of trajectories
        p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

        logger.info(
            f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
        logger.info(
            f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}'
        )
        logger.info(
            f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
        self.trajectories = trajectories
        self.num_trajectories = num_trajectories
        self.p_sample = p_sample
        self.sorted_inds = sorted_inds
        self.state_dim = trajectories[0]['observations'].shape[1]
        self.act_dim = trajectories[0]['actions'].shape[1]

    def get_batch(self, batch_size=256, max_len=20):
        batch_inds = np.random.choice(
            np.arange(self.num_trajectories),
            size=batch_size,
            replace=True,
            p=self.p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = self.trajectories[int(self.sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(
                1, -1, self.state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(
                1, -1, self.act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= self.
                          max_ep_len] = self.max_ep_len - 1  # padding cutoff
            rtg.append(
                self.discount_cumsum(traj['rewards'][si:],
                                     gamma=1.)[:s[-1].shape[1] + 1].reshape(
                                         1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate(
                    [rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen, self.state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - self.state_mean) / self.state_std
            a[-1] = np.concatenate(
                [np.ones((1, max_len - tlen, self.act_dim)) * -10., a[-1]],
                axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]],
                                   axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]],
                                   axis=1)
            rtg[-1] = np.concatenate(
                [np.zeros(
                    (1, max_len - tlen, 1)), rtg[-1]], axis=1) / self.scale
            timesteps[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(
                np.concatenate(
                    [np.zeros((1, max_len - tlen)),
                     np.ones((1, tlen))],
                    axis=1))

        device = self.device
        s = torch.from_numpy(np.concatenate(s, axis=0)).to(
            dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(
            dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(
            dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(
            dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(
            dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(
            dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask

    def discount_cumsum(self, x, gamma):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum
