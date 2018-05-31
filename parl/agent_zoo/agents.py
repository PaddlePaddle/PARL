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

import numpy as np
from parl.common.logging import GameLogEntry
from parl.framework.agent import AgentBase


class SimpleRLAgent(AgentBase):
    """
    This class serves as a template of simple RL algorithms, which has only one
    ComputationTask, "RL", i.e., using and learning an RL policy. 
    
    By using different AgentHelpers, this Agent can be applied to either on-
    policy or off-policy RL algorithms.
    """

    def __init__(self, env, num_games):
        super(SimpleRLAgent, self).__init__(env, num_games)

    def _run_one_episode(self):
        # sensor_inputs, (prev_)states and actions are all dict
        max_steps = self.env._max_episode_steps
        obs = self.env.reset()
        episode_end = False
        r = 0
        log_entry = GameLogEntry(self.id, 'RL')
        for t in range(max_steps - 1):
            #self.env.render()
            sensor = np.array(obs).astype("float32")
            inputs = dict(sensor=sensor)
            actions, _ = self.helpers['RL'].predict(inputs)
            try:
                a = actions["action"][0][0]
                next_obs, r, next_episode_end, _ = self.env.step(a)
            except Exception:
                print actions
                raise Exception

            r /= 100.0
            log_entry.num_steps += 1
            log_entry.total_reward += r
            data = {}
            data.update(inputs)
            data.update({
                "action": np.array([a]).astype("int64"),
                "reward": np.array([r]).astype("float32"),
                "episode_end": np.array([episode_end]).astype("uint8")
            })
            self.helpers['RL'].store_data(data)
            obs = next_obs
            episode_end = next_episode_end
            if next_episode_end:
                data = dict(
                    sensor=np.array(next_obs).astype("float32"),
                    action=np.array([0]).astype("int64"),
                    reward=np.array([0]).astype("float32"),
                    episode_end=np.array([episode_end]).astype("uint8"))

                self.helpers['RL'].store_data(data)
                break
        return log_entry.total_reward
        #self.log_q.put(log_entry)
