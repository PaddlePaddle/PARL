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

import grpc
import json
import numpy as np
import simulator_pb2
import simulator_pb2_grpc
from args import get_client_args
from env_wrapper import FrameSkip, ActionScale, PelvisBasedObs, MAXTIME_LIMIT, CustomR2Env, RunFastestReward, FixedTargetSpeedReward, Round2Reward
from osim.env import ProstheticsEnv
from parl.utils import logger

ProstheticsEnv.time_limit = MAXTIME_LIMIT


class Worker(object):
    def __init__(self, server_ip='localhost', server_port=5007):
        if args.ident is not None:
            self.worker_id = args.ident
        else:
            self.worker_id = np.random.randint(int(1e18))

        self.address = '{}:{}'.format(server_ip, server_port)

        random_seed = int(self.worker_id % int(1e9))
        np.random.seed(random_seed)

        env = ProstheticsEnv(visualize=False, seed=random_seed)
        env.change_model(
            model='3D', difficulty=1, prosthetic=True, seed=random_seed)
        env.spec.timestep_limit = MAXTIME_LIMIT
        env = CustomR2Env(env)

        if args.reward_type == 'RunFastest':
            env = RunFastestReward(env)
        elif args.reward_type == 'FixedTargetSpeed':
            env = FixedTargetSpeedReward(
                env, args.target_v, args.act_penalty_lowerbound,
                args.act_penalty_coeff, args.vel_penalty_coeff)
        elif args.reward_type == 'Round2':
            env = Round2Reward(env, args.act_penalty_lowerbound,
                               args.act_penalty_coeff, args.vel_penalty_coeff)
        else:
            assert False, 'Not supported reward type!'

        env = FrameSkip(env, 4)
        env = ActionScale(env)
        self.env = PelvisBasedObs(env)

    def run(self):
        observation = self.env.reset(project=False, stage=args.stage)
        reward = 0
        done = False
        info = {'shaping_reward': 0.0}
        info['first'] = True
        with grpc.insecure_channel(self.address) as channel:
            stub = simulator_pb2_grpc.SimulatorStub(channel)
            while True:
                response = stub.Send(
                    simulator_pb2.Request(
                        observation=observation,
                        reward=reward,
                        done=done,
                        info=json.dumps(info),
                        id=self.worker_id))

                extra = json.loads(response.extra)

                if 'reset' in extra and extra['reset']:
                    logger.info('Server require to reset!')
                    observation = self.env.reset(
                        project=False, stage=args.stage)
                    reward = 0
                    done = False
                    info = {'shaping_reward': 0.0}
                    continue

                if 'shutdown' in extra and extra['shutdown']:
                    break

                action = np.array(response.action)
                next_observation, reward, done, info = self.env.step(
                    action, project=False)

                # debug info
                if args.debug:
                    logger.info("dim:{} obs:{}   act:{}   reward:{}  done:{}  info:{}".format(\
                        len(observation), np.sum(observation), np.sum(action), reward, done, info))
                observation = next_observation
                if done:
                    observation = self.env.reset(
                        project=False, stage=args.stage)

                    # the last observation should be used to compute append_value in simulator_server
                    info['last_obs'] = next_observation.tolist()


if __name__ == '__main__':
    args = get_client_args()

    worker = Worker(server_ip=args.ip, server_port=args.port)

    worker.run()
