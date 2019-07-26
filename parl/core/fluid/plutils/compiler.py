#!/usr/bin/env python
# coding=utf8
# File: compiler.py
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
    if loss is not None:
        assert isinstance(
            loss, fluid.framework.
            Variable), 'type of loss is expected to be a fluid tensor'
    # TODO: after solving the learning rate issue that occurs in training A2C algorithm, set it to 3.
    os.environ['CPU_NUM'] = '1'
    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_threads = 3 * 4

    build_strategy = fluid.BuildStrategy()
    build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce

    return fluid.compiler.CompiledProgram(program).with_data_parallel(
        loss_name=loss.name,
        exec_strategy=exec_strategy,
        build_strategy=build_strategy)
