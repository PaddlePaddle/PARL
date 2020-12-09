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

import torch
import os
from grid2op.Runner import Runner
from l2rpn_baselines.utils.save_log_gif import save_log_gif
from rl_agent import RLAgent
import argparse


def cli():
    parser = argparse.ArgumentParser(description="Eval baseline RLAgent")
    parser.add_argument(
        "--load_path",
        default='./saved_files',
        help="The path to the model [.h5]")
    parser.add_argument(
        "--logs_path",
        required=False,
        default='./logs_path',
        type=str,
        help="Path to output logs directory")
    parser.add_argument(
        "--nb_episode",
        required=False,
        default=1,
        type=int,
        help="Number of episodes to evaluate")
    parser.add_argument(
        "--nb_process",
        required=False,
        default=1,
        type=int,
        help="Number of cores to use")
    parser.add_argument(
        "--max_steps",
        required=False,
        default=-1,
        type=int,
        help="Maximum number of steps per scenario")
    parser.add_argument(
        "--save_gif", action='store_true', help="Enable GIF Output")
    parser.add_argument(
        "--verbose", action='store_true', help="Verbose runner output")
    return parser.parse_args()


def evaluate(env,
             load_path="saved_files",
             logs_path=None,
             nb_episode=1,
             nb_process=1,
             max_steps=-1,
             verbose=False,
             save_gif=False,
             **kwargs):
    """
    In order to submit a valid basline, it is mandatory to provide a "evaluate" function with the same signature as this one.

    Parameters
    ----------
    env: :class:`grid2op.Environment.Environment`
        The environment on which the baseline will be evaluated.

    load_path: ``str``
        The path where the model is stored. This is used by the agent when calling "agent.load)

    logs_path: ``str``
        The path where the agents results will be stored.

    nb_episode: ``int``
        Number of episodes to run for the assessment of the performance.
        By default it's 1.

    nb_process: ``int``
        Number of process to be used for the assessment of the performance.
        Should be an integer greater than 1. By defaults it's 1.

    max_steps: ``int``
        Maximum number of timestep each episode can last. It should be a positive integer or -1.
        -1 means that the entire episode is run (until the chronics is out of data or until a game over).
        By default it's -1.

    verbose: ``bool``
        verbosity of the output

    save_gif: ``bool``
        Whether or not to save a gif into each episode folder corresponding to the representation of the said episode.

    kwargs:
        Other key words arguments that you are free to use for either building the agent save it etc.

    Returns
    -------
    ``None``
    """
    runner_params = env.get_params_for_runner()
    runner_params["verbose"] = verbose

    # Create the agent (this piece of code can change)
    agent = RLAgent(env.action_space)

    # Load weights from file (for example)
    agent.load(load_path)

    # Build runner
    runner = Runner(**runner_params, agentClass=None, agentInstance=agent)

    # you can do stuff with your model here

    # start the runner
    res = runner.run(
        path_save=logs_path,
        nb_episode=nb_episode,
        nb_process=nb_process,
        max_iter=max_steps,
        pbar=False)

    # Print summary
    print("Evaluation summary:")
    for _, chron_name, cum_reward, nb_time_step, max_ts in res:
        msg_tmp = "\tFor chronics located at {}\n".format(chron_name)
        msg_tmp += "\t\t - cumulative reward: {:.6f}\n".format(cum_reward)
        msg_tmp += "\t\t - number of time steps completed: {:.0f} / {:.0f}".format(
            nb_time_step, max_ts)
        print(msg_tmp)

    if save_gif:
        save_log_gif(logs_path, res)


if __name__ == "__main__":
    """
    This is a possible implementation of the eval script.
    """
    import grid2op
    from lightsim2grid.LightSimBackend import LightSimBackend
    args = cli()
    backend = LightSimBackend()
    env = grid2op.make('l2rpn_neurips_2020_track2_small', backend=backend)
    evaluate(
        env,
        load_path=args.load_path,
        logs_path=args.logs_path,
        nb_episode=args.nb_episode,
        nb_process=args.nb_process,
        max_steps=args.max_steps,
        verbose=args.verbose,
        save_gif=args.save_gif)
