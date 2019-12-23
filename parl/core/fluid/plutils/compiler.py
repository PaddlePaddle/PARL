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

import paddle.fluid as fluid
import os


def compile(program, loss=None):
    """ transfer the program into a new program that runs in multi-cpus or multi-gpus.
    This function uses the `fluid.compiler.CompiledProgram` to transfer the program.
    For more detail about speeding the program, please visit 
    "https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/user_guides/howto/training/single_node.html#id7"

    Args:
        program(fluid.Program): a normal fluid program.
        loss_name(str): Optional. The loss tensor of a trainable program. Set it to None if you are transferring a prediction or evaluation program.
    """
    loss_name = None
    if loss is not None:
        assert isinstance(
            loss, fluid.framework.
            Variable), 'type of loss is expected to be a fluid tensor'
        loss_name = loss.name
    # TODO: after solving the learning rate issue that occurs in training A2C algorithm, set it to 3.
    os.environ['CPU_NUM'] = '1'
    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_threads = 3 * 4

    build_strategy = fluid.BuildStrategy()
    build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce

    compiled_program = fluid.compiler.CompiledProgram(
        program).with_data_parallel(
            loss_name=loss_name,
            exec_strategy=exec_strategy,
            build_strategy=build_strategy)
    compiled_program._init_program = program
    return compiled_program
