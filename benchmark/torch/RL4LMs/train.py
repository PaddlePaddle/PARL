#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import sys
from argparse import ArgumentParser
import datetime
import yaml
import collections
from parl.utils import logger
import torch
import time

# instructor and reward function
from instructor import InstructorGroup

# evaluation, metrics, tokenizer & dataset
from rl4lms_utils import build_metrics, build_tokenizer, build_datapool
from rl4lms_utils import Examiner

# rollout
from rl4lms_utils import DictRolloutBuffer, RolloutUtil

# agent, algorithm and model
from rl4lms_ppo import RL4LMsPPO
from rl4lms_agent import RL4LMsAgent
from seq2seq_model import Seq2SeqLMModel


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = build_tokenizer(config["tokenizer"])

    # datapool
    samples_by_split = build_datapool(config["datapool"])

    instructor_group = InstructorGroup(
        instructor_config=config["instructor"],
        tokenizer=tokenizer,
        tokenizer_config=config["tokenizer"],
        datapool_config=config["datapool"],
    )

    model_config = config["agent"]["alg"]["model"]
    rl4lms_model = Seq2SeqLMModel(
        observation_space=instructor_group.observation_space,
        action_space=instructor_group.action_space,
        device=device,
        model_name=model_config["args"]["model_name"],
        apply_model_parallel=model_config["args"]["apply_model_parallel"],
        prompt_truncation_side=model_config["args"]["prompt_truncation_side"],
        generation_kwargs=model_config["args"]["generation_kwargs"])
    alg_config = config["agent"]["alg"]
    rl4lm_alg = RL4LMsPPO(
        model=rl4lms_model,
        initial_lr=alg_config["args"]["initial_lr"],
        entropy_coef=alg_config["args"]["entropy_coef"])
    agent_config = config["agent"]
    agent = RL4LMsAgent(rl4lm_alg,
                        n_epochs=agent_config["args"]["n_epochs"],
                        batch_size=agent_config["args"]["batch_size"],)

    buffer_config = config["rollout_buffer"]
    rollout_buffer = DictRolloutBuffer(
        buffer_size= buffer_config["args"]["n_steps_per_episode"] * instructor_group.n_instructors,
        observation_space=instructor_group.observation_space,
        action_space=instructor_group.action_space,
        device=device,
    )
    rollout_util = RolloutUtil(config["kl_div"])

    n_iters = int(config["train_evaluation"]["n_iters"])
    n_steps_per_iter = instructor_group.n_instructors * buffer_config["args"]["n_steps_per_episode"]

    # gen kwargs for evaluation
    examiner_config = config["examiner"]
    # metrics
    metrics = build_metrics(examiner_config["metrics"])
    examiner = Examiner(
        tokenizer=tokenizer,
        eval_batch_size=examiner_config["args"]["eval_batch_size"],
        max_prompt_length=examiner_config["args"]["max_prompt_length"],
        eval_gen_kwargs=examiner_config["args"]["generation_kwargs"],
        metrics=metrics,
        samples_by_split=samples_by_split,
    )

    iter_start = 0
    examiner.evaluate(policy=agent.alg.model, sample_name_list=["val", "test"], epoch=iter_start)

    epoch = 0
    for epoch in range(iter_start, n_iters):
        print("========== BEGIN ==========")
        print(f"outer epoch: {epoch} / {n_iters - 1}")
        print("========== BEGIN ==========")
        outer_start_time = time.time()

        num_timesteps = 0

        while num_timesteps < n_steps_per_iter:
            run_timesteps = rollout_util.collect_rollouts(agent, instructor_group, rollout_buffer)
            num_timesteps += run_timesteps
            agent.learn(rollout_buffer)

        outer_end_time = time.time()
        print("========== END ==========")
        print(f"outer epoch: {epoch} / {n_iters - 1}")
        print(f"time used: {outer_end_time - outer_start_time} second(s), left time:"
              f"  {1.0 * (outer_end_time - outer_start_time) * (n_iters - epoch - 1) / 60 / 60} hour(s)")
        print("========== END ==========")

        # evaluate on val set in the given intervals
        if (epoch + 1) % config["train_evaluation"]["eval_every"] == 0:
            examiner.evaluate(policy=agent.alg.model, sample_name_list=["val"], epoch=epoch)

    examiner.evaluate(policy=agent.alg.model, sample_name_list=["val", "test"], epoch=epoch)


if __name__ == '__main__':
    parser = ArgumentParser(description="Fine-tune LM to generate controlled text")
    parser.add_argument("--config_path", type=str, help="path to the config file")
    parser.add_argument("--project_name", type=str, help="project name", default="rl4lm_exps")
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="experiment name",
        default="rl4lm_experiment",
    )
    parser.add_argument(
        "--base_path_to_store_results",
        type=str,
        help="Base path to store experiment results",
        default=os.getcwd(),
    )
    parser.add_argument("--entity_name", type=str, help="entity name", default="summarization")
    args = parser.parse_args()

    # load the config file
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)

    recursive_dict_update(config, vars(args))
    log_dir = f"./{args.project_name}/{args.experiment_name}/{args.entity_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    logger.set_dir(log_dir)
    config["logging_dir"] = log_dir
    config["sys_arg"] = sys.argv
    logger.info(config)
    logger.set_level("DEBUG")

    main(config)
