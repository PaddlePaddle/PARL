# Third Party Code
# The following code is copied or modified from:
# https://github.com/kzl/decision-transformer/tree/master/gym/decision_transformer

import numpy as np
import torch
from tqdm import tqdm


def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        agent,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
):

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(
        device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(
        ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat(
            [actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = agent.predict(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(
            1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        pred_return = target_return[0, -1] - (reward / scale)
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat([
            timesteps,
            torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)
        ],
                              dim=1)

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length


def eval_episodes(target_rew, env, state_dim, act_dim, agent, max_ep_len,
                  scale, state_mean, state_std, device):
    returns, lengths = [], []
    num_eval_episodes = 100
    for _ in tqdm(range(num_eval_episodes), desc='eval'):
        with torch.no_grad():
            ret, length = evaluate_episode_rtg(
                env,
                state_dim,
                act_dim,
                agent,
                max_ep_len=max_ep_len,
                scale=scale,
                target_return=target_rew / scale,
                state_mean=state_mean,
                state_std=state_std,
                device=device,
            )
        returns.append(ret)
        lengths.append(length)
    return {
        f'target_{target_rew}_return_mean': np.mean(returns),
        f'target_{target_rew}_return_std': np.std(returns),
        f'target_{target_rew}_length_mean': np.mean(lengths),
        f'target_{target_rew}_length_std': np.std(lengths),
    }
