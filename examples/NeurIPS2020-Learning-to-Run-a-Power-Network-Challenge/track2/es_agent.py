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

import parl
import paddle.fluid as fluid
from parl import layers
import numpy as np


class ESAgent(parl.Agent):
    def __init__(self, algorithm):
        super(ESAgent, self).__init__(algorithm)

    def build_program(self):
        self.predict_program = fluid.Program()

        with fluid.program_guard(self.predict_program):
            obs = layers.data(name='obs', shape=[639], dtype='float32')
            self.predicted_unitary_actions_rho = self.alg.predict(obs)

    def predict_unitary_actions_rho(self, observation):
        processed_obs = observation.reshape(1, -1)
        predicted_rho = self.fluid_executor.run(
            self.predict_program,
            feed={'obs': processed_obs},
            fetch_list=[self.predicted_unitary_actions_rho])[0].reshape(-1)
        sorted_idx = np.argsort(predicted_rho)
        return sorted_idx.tolist(), predicted_rho

    def restore(self, save_path, filename):
        fluid.io.load_params(self.fluid_executor, save_path,
                             self.predict_program, filename)
