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

import parl
import numpy as np
from osim.env import L2M2019Env
from env_wrapper import FrameSkip, ActionScale, OfficialObs, FinalReward, FirstTarget


@parl.remote_class
class Actor(object):
    def __init__(self,
                 difficulty,
                 vel_penalty_coeff,
                 muscle_penalty_coeff,
                 penalty_coeff,
                 only_first_target=False):

        random_seed = np.random.randint(int(1e9))

        env = L2M2019Env(
            difficulty=difficulty, visualize=False, seed=random_seed)
        max_timelimit = env.time_limit

        env = FinalReward(
            env,
            max_timelimit=max_timelimit,
            vel_penalty_coeff=vel_penalty_coeff,
            muscle_penalty_coeff=muscle_penalty_coeff,
            penalty_coeff=penalty_coeff)

        if only_first_target:
            assert difficulty == 3, "argument `only_first_target` is available only in `difficulty=3`."
            env = FirstTarget(env)

        env = FrameSkip(env)
        env = ActionScale(env)
        self.env = OfficialObs(env, max_timelimit=max_timelimit)

    def reset(self):
        observation = self.env.reset(project=False)
        return observation

    def step(self, action):
        return self.env.step(action, project=False)
