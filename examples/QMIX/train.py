#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from env_wrapper import SC2EnvWrapper
from replay_buffer import EpisodeExperience, EpisodeReplayBuffer
from qmixer_model import QMixerModel
from rnn_model import RNNModel
from parl.algorithms import QMIX
from qmix_agent import QMixAgent
import parl
from parl.utils import logger
from parl.utils import summary
import numpy as np
from copy import deepcopy
from qmix_config import QMixConfig

logger.set_dir('./log_path')


def run_train_episode(env, agent, rpm, config):
    episode_limit = config['episode_limit']
    agent.reset_agent()
    episode_reward = 0.0
    episode_step = 0
    terminated = False
    state, obs = env.reset()

    episode_experience = EpisodeExperience(episode_limit)

    while not terminated:
        available_actions = env.get_available_actions()
        actions = agent.sample(obs, available_actions)
        #next_state, next_obs, reward, terminated = env.step(actions)
        next_state, next_obs, reward, terminated = env.step(actions)
        episode_reward += reward
        episode_step += 1
        episode_experience.add(state, actions, [reward], [terminated], obs,
                               available_actions, [0])
        state = next_state
        obs = next_obs

    # fill the episode
    state_zero = np.zeros_like(state, dtype=state.dtype)
    actions_zero = np.zeros_like(actions, dtype=actions.dtype)
    obs_zero = np.zeros_like(obs, dtype=obs.dtype)
    available_actions_zero = np.zeros_like(
        available_actions, dtype=available_actions.dtype)
    reward_zero = 0
    terminated_zero = True
    for _ in range(episode_step, episode_limit):
        episode_experience.add(state_zero, actions_zero, [reward_zero],
                               [terminated_zero], obs_zero,
                               available_actions_zero, [1])
    rpm.add(episode_experience)
    is_win = env.win_counted

    mean_loss = []
    mean_td_error = []
    if rpm.count > config['memory_warmup_size']:
        for _ in range(2):
            s_batch, a_batch, r_batch, t_batch, obs_batch, available_actions_batch,\
                    filled_batch = rpm.sample_batch(config['batch_size'])
            loss, td_error = agent.learn(s_batch, a_batch, r_batch, t_batch,
                                         obs_batch, available_actions_batch,
                                         filled_batch)
            mean_loss.append(loss)
            mean_td_error.append(td_error)

    mean_loss = np.mean(mean_loss) if mean_loss else None
    mean_td_error = np.mean(mean_td_error) if mean_td_error else None
    return episode_reward, episode_step, is_win, mean_loss, mean_td_error


def run_evaluate_episode(env, agent):
    agent.reset_agent()
    episode_reward = 0.0
    episode_step = 0
    terminated = False
    state, obs = env.reset()

    while not terminated:
        available_actions = env.get_available_actions()
        actions = agent.predict(obs, available_actions)
        state, obs, reward, terminated = env.step(actions)
        episode_step += 1
        episode_reward += reward

    is_win = env.win_counted
    return episode_reward, episode_step, is_win


def main():
    config = deepcopy(QMixConfig)
    env = StarCraft2Env(
        map_name=config['scenario'], difficulty=config['difficulty'])
    env = SC2EnvWrapper(env)
    config['episode_limit'] = env.episode_limit
    config['obs_shape'] = env.obs_shape
    config['state_shape'] = env.state_shape
    config['n_agents'] = env.n_agents
    config['n_actions'] = env.n_actions

    rpm = EpisodeReplayBuffer(config['replay_buffer_size'])
    agent_model = RNNModel(config['obs_shape'], config['n_actions'],
                           config['rnn_hidden_dim'])
    qmixer_model = QMixerModel(
        config['n_agents'], config['state_shape'], config['mixing_embed_dim'],
        config['hypernet_layers'], config['hypernet_embed_dim'])

    algorithm = QMIX(agent_model, qmixer_model, config['double_q'],
                     config['gamma'], config['lr'], config['clip_grad_norm'])

    qmix_agent = QMixAgent(
        algorithm, config['exploration_start'], config['min_exploration'],
        config['exploration_decay'], config['update_target_interval'])

    while rpm.count < config['memory_warmup_size']:
        train_reward, train_step, train_is_win, train_loss, train_td_error\
                = run_train_episode(env, qmix_agent, rpm, config)

    total_steps = 0
    last_test_step = -1e10
    while total_steps < config['training_steps']:
        train_reward, train_step, train_is_win, train_loss, train_td_error\
                = run_train_episode(env, qmix_agent, rpm, config)
        total_steps += train_step

        if total_steps - last_test_step >= config['test_steps']:
            last_test_step = total_steps
            eval_is_win_buffer = []
            eval_reward_buffer = []
            eval_steps_buffer = []
            for _ in range(3):
                eval_reward, eval_step, eval_is_win = run_evaluate_episode(
                    env, qmix_agent)
                eval_reward_buffer.append(eval_reward)
                eval_steps_buffer.append(eval_step)
                eval_is_win_buffer.append(eval_is_win)

            summary.add_scalar('train_loss', train_loss, total_steps)
            summary.add_scalar('eval_reward', np.mean(eval_reward_buffer),
                               total_steps)
            summary.add_scalar('eval_steps', np.mean(eval_steps_buffer),
                               total_steps)
            summary.add_scalar('eval_win_rate', np.mean(eval_is_win_buffer),
                               total_steps)
            summary.add_scalar('exploration', qmix_agent.exploration,
                               total_steps)
            summary.add_scalar('replay_buffer_size', rpm.count, total_steps)
            summary.add_scalar('target_update_count',
                               qmix_agent.target_update_count, total_steps)
            summary.add_scalar('train_td_error:', train_td_error, total_steps)


if __name__ == '__main__':
    main()
