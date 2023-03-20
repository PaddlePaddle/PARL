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

import sys
from t5_ppo_config import config
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

    agent_config = config["agent"]
    model_config = agent_config["model"]
    rl4lms_model = Seq2SeqLMModel(
        observation_space=instructor_group.observation_space,
        action_space=instructor_group.action_space,
        device=device,
        model_name=model_config["model_name"],
        apply_model_parallel=model_config["apply_model_parallel"],
        prompt_truncation_side=model_config["prompt_truncation_side"],
        generation_kwargs=model_config["generation_kwargs"])
    alg_config = agent_config["alg"]
    rl4lm_alg = RL4LMsPPO(
        model=rl4lms_model,
        initial_lr=alg_config["initial_lr"],
        entropy_coef=alg_config["entropy_coef"])
    agent = RL4LMsAgent(
        rl4lm_alg,
        n_epochs=agent_config["n_epochs"],
        batch_size=agent_config["batch_size"],
    )

    buffer_config = config["rollout_buffer"]
    rollout_buffer = DictRolloutBuffer(
        buffer_size=buffer_config["n_steps_per_instructor"] * instructor_group.n_instructors,
        observation_space=instructor_group.observation_space,
        action_space=instructor_group.action_space,
        device=device,
    )
    rollout_util = RolloutUtil(config["kl_div"])

    n_iters = int(config["train_evaluation"]["n_iters"])
    n_steps_per_iter = instructor_group.n_instructors * buffer_config["n_steps_per_instructor"]

    # gen kwargs for evaluation
    examiner_config = config["examiner"]
    # metrics
    metrics = build_metrics(examiner_config["metrics"])
    examiner = Examiner(
        tokenizer=tokenizer,
        eval_batch_size=examiner_config["eval_batch_size"],
        max_prompt_length=examiner_config["max_prompt_length"],
        eval_gen_kwargs=examiner_config["generation_kwargs"],
        metrics=metrics,
        samples_by_split=samples_by_split,
    )

    iter_start = 0
    examiner.evaluate(policy=agent.alg.model, sample_name_list=["val", "test"], epoch=iter_start)

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

    # during training, we evaluate on VALIDATION set, and finally we evaluate on TEST set
    examiner.evaluate(policy=agent.alg.model, sample_name_list=["test"], epoch=epoch)


if __name__ == '__main__':
    logger.auto_set_dir()

    config["logging_dir"] = logger.get_dir()
    config["sys_arg"] = sys.argv

    logger.info(config)

    main(config)
