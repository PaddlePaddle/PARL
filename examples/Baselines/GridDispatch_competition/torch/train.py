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

import os
os.environ['PARL_BACKEND'] = 'torch'

import numpy as np
import argparse
import threading
import time
import parl
from parl.utils import logger, tensorboard, ReplayMemory
from grid_model import GridModel
from grid_agent import GridAgent
from parl.algorithms import SAC
from env_wrapper import get_env

WARMUP_STEPS = 1e4
MEMORY_SIZE = int(1e6)
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
OBS_DIM = 819
ACT_DIM = 54


@parl.remote_class
class Actor(object):
    def __init__(self, args):
        self.env = get_env()

        obs_dim = OBS_DIM
        action_dim = ACT_DIM
        self.action_dim = action_dim

        # Initialize model, algorithm, agent, replay_memory
        model = GridModel(obs_dim, action_dim)
        algorithm = SAC(
            model,
            gamma=GAMMA,
            tau=TAU,
            alpha=args.alpha,
            actor_lr=ACTOR_LR,
            critic_lr=CRITIC_LR)
        self.agent = GridAgent(algorithm)

    def sample(self, weights, random_action):
        self.agent.set_weights(weights)

        obs = self.env.reset()

        done = False
        episode_reward, episode_steps = 0, 0
        sample_data = []
        while not done:
            # Select action randomly or according to policy
            if random_action:
                action = np.random.uniform(-1, 1, size=self.action_dim)
            else:
                action = self.agent.sample(obs)

            # Perform action
            next_obs, reward, done, info = self.env.step(action)
            terminal = done and not info['timeout']
            terminal = float(terminal)

            sample_data.append((obs, action, reward, next_obs, terminal))

            obs = next_obs
            episode_reward += info['origin_reward']
            episode_steps += 1

        return sample_data, episode_steps, episode_reward


class Learner(object):
    def __init__(self, args):
        self.model_lock = threading.Lock()
        self.rpm_lock = threading.Lock()
        self.log_lock = threading.Lock()

        self.args = args

        obs_dim = OBS_DIM
        action_dim = ACT_DIM

        # Initialize model, algorithm, agent, replay_memory
        model = GridModel(obs_dim, action_dim)
        algorithm = SAC(
            model,
            gamma=GAMMA,
            tau=TAU,
            alpha=args.alpha,
            actor_lr=ACTOR_LR,
            critic_lr=CRITIC_LR)
        self.agent = GridAgent(algorithm)

        self.agent.restore("./torch_pretrain_model")

        self.rpm = ReplayMemory(
            max_size=MEMORY_SIZE, obs_dim=obs_dim, act_dim=action_dim)

        self.total_steps = 0
        self.total_MDP_steps = 0
        self.save_cnt = 0

        parl.connect(
            args.xparl_addr,
            distributed_files=[
                'lib64/*',
                'model_jm/*',
                'Agent/*',
                'Environment/*',
                'Observation/*',
                'Reward/*',
                'utilize/*',
            ])
        for _ in range(args.actor_num):
            th = threading.Thread(target=self.run_sampling)
            th.setDaemon(True)
            th.start()

    def run_sampling(self):
        actor = Actor(self.args)
        while True:
            start = time.time()
            weights = None
            with self.model_lock:
                weights = self.agent.get_weights()

            random_action = False
            if self.rpm.size() < WARMUP_STEPS:
                random_action = True

            sample_data, episode_steps, episode_reward = actor.sample(
                weights, random_action)

            # Store data in replay memory
            with self.rpm_lock:
                for data in sample_data:
                    self.rpm.append(*data)

            sample_time = time.time() - start
            start = time.time()

            critic_loss, actor_loss = None, None
            # Train agent after collecting sufficient data
            if self.rpm.size() >= WARMUP_STEPS:
                for _ in range(len(sample_data)):
                    with self.rpm_lock:
                        batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = self.rpm.sample_batch(
                            BATCH_SIZE)
                    with self.model_lock:
                        critic_loss, actor_loss = self.agent.learn(
                            batch_obs, batch_action, batch_reward,
                            batch_next_obs, batch_terminal)
            learn_time = time.time() - start

            mean_action = np.mean(
                np.array([x[1] for x in sample_data]), axis=0)

            with self.log_lock:
                self.total_steps += episode_steps
                self.total_MDP_steps += len(sample_data)
                tensorboard.add_scalar('train/episode_reward', episode_reward,
                                       self.total_steps)
                tensorboard.add_scalar('train/episode_steps', episode_steps,
                                       self.total_steps)
                if critic_loss is not None:
                    tensorboard.add_scalar('train/critic_loss', critic_loss,
                                           self.total_steps)
                    tensorboard.add_scalar('train/actor_loss', actor_loss,
                                           self.total_steps)
                logger.info('Total Steps: {} Reward: {} Steps: {}'.format(
                    self.total_steps, episode_reward, episode_steps))

                if self.total_steps // self.args.save_every_steps >= self.save_cnt:
                    while self.total_steps // self.args.save_every_steps >= self.save_cnt:
                        self.save_cnt += 1
                    with self.model_lock:
                        self.agent.save(
                            os.path.join(self.args.save_dir,
                                         "model-{}".format(self.total_steps)))


def main():
    learner = Learner(args)

    while True:
        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_every_steps', type=int, default=10000)
    parser.add_argument(
        "--alpha",
        default=0.2,
        type=float,
        help=
        'Determines the relative importance of entropy term against the reward'
    )
    parser.add_argument('--xparl_addr', type=str, default="localhost:8010")
    parser.add_argument('--actor_num', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default="./saved_models")
    args = parser.parse_args()

    main()
