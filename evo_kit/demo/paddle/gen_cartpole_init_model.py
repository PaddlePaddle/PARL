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

from paddle import fluid


def net(obs, act_dim):
    hid1_size = act_dim * 10
    hid1 = fluid.layers.fc(obs, size=hid1_size)
    prob = fluid.layers.fc(hid1, size=act_dim, act='softmax')
    return prob


if __name__ == '__main__':
    obs_dim = 4
    act_dim = 2

    obs = fluid.layers.data(name="obs", shape=[obs_dim], dtype='float32')

    prob = net(obs, act_dim)

    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())
    fluid.io.save_inference_model(
        dirname='cartpole_init_model',
        feeded_var_names=['obs'],
        target_vars=[prob],
        params_filename='params',
        model_filename='model',
        executor=exe)
