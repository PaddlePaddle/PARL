import os
import sys
from argparse import ArgumentParser
import datetime
import yaml
import collections
from parl.utils import logger

import torch
import time

# env and reward function
from rl4lms_utils import build_reward_fn
from reviewer import ReviewerGroup

# evaluation, metrics, tokenizer & dataset
from rl4lms_utils import build_metrics, build_tokenizer, build_datapool
from rl4lms_utils import evaluate_on_samples

# rollout
from rl4lms_utils import DictRolloutBuffer, RolloutUtil

# agent, algorithm and model
from rl4lm_ppo import RL4LMPPO
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
    device = torch.device("cuda" if torch.cuda.
                                   is_available() else "cpu")

    tokenizer = build_tokenizer(config["tokenizer"])

    # reward function & metrics
    reward_fn = build_reward_fn(config["reward_fn"])
    metrics = build_metrics(config["train_evaluation"]["metrics"])

    # datapool
    samples_by_split = build_datapool(config["datapool"])


    reviewer_group = ReviewerGroup(reviewer_config=config["reviewer"],
                                   reward_fn=reward_fn,
                                   tokenizer=tokenizer,
                                   question_samples=samples_by_split["train"])

    rl4lms_model = Seq2SeqLMModel(
        observation_space = reviewer_group.observation_space,
        action_space= reviewer_group.action_space,
        device=device,
        **config["alg"]["model"]["args"]
    )
    rl4lm_alg = RL4LMPPO(model=rl4lms_model, device=device, **config["alg"]["args"])
    agent = RL4LMsAgent(rl4lm_alg, config["alg"])

    rollout_buffer = DictRolloutBuffer(
        buffer_size=agent.alg.n_steps * reviewer_group.n_reviewers,
        observation_space=reviewer_group.observation_space,
        action_space=reviewer_group.action_space,
        device=device,
        gamma=agent.alg.gamma,
        gae_lambda=agent.alg.gae_lambda,
    )
    rollout_util = RolloutUtil(config["alg"]["kl_div"], reviewer_group)

    n_iters = int(config["train_evaluation"]["n_iters"])
    n_steps_per_iter = reviewer_group.n_reviewers * agent.alg.n_steps

    max_prompt_length = config["reviewer"]["args"]["max_prompt_length"]

    # gen kwargs for evaluation
    eval_gen_kwargs = config["train_evaluation"]["generation_kwargs"]
    eval_batch_size = config["train_evaluation"]["eval_batch_size"]
    eval_splits = ["val", "test"]

    iter_start = 0
    for sp in eval_splits:
        evaluate_on_samples(policy=agent.alg.model,
                            tokenizer=tokenizer,
                            samples=samples_by_split[sp],
                            batch_size=eval_batch_size,
                            max_prompt_length=max_prompt_length,
                            metrics=metrics,
                            epoch=iter_start,
                            split_name=sp,
                            gen_kwargs=eval_gen_kwargs)
    epoch = 0
    for epoch in range(iter_start, n_iters):
        print("========== BEGIN ==========")
        print(f"outer epoch: {epoch} / {n_iters - 1}")
        print("========== BEGIN ==========")
        outer_start_time = time.time()

        num_timesteps = 0

        while num_timesteps < n_steps_per_iter:
            run_timesteps = rollout_util.collect_rollouts(agent, reviewer_group, rollout_buffer, device)
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
            evaluate_on_samples(policy=agent.alg.model,
                                tokenizer=tokenizer,
                                samples=samples_by_split["val"],
                                batch_size=eval_batch_size,
                                max_prompt_length=max_prompt_length,
                                metrics=metrics,
                                epoch=epoch,
                                split_name="val",
                                gen_kwargs=eval_gen_kwargs)


    for sp in eval_splits:
        evaluate_on_samples(policy=agent.alg.model,
                            tokenizer=tokenizer,
                            samples=samples_by_split[sp],
                            batch_size=eval_batch_size,
                            max_prompt_length=max_prompt_length,
                            metrics=metrics,
                            epoch=epoch,
                            split_name=sp,
                            gen_kwargs=eval_gen_kwargs)


if __name__ == '__main__':
    parser = ArgumentParser(description="Fine-tune LM to generate controlled text")
    parser.add_argument("--config_path", type=str, help="path to the config file")
    parser.add_argument(
        "--project_name", type=str, help="project name", default="rl4lm_exps"
    )
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
    parser.add_argument(
        "--entity_name", type=str, help="entity name", default="summarization"
    )
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

