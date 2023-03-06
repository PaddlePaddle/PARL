import os
import time
from functools import partial
from typing import Any, Dict, List
import numpy as np


from benchmark.torch.RL4LMs.utils import Sample, RewardFunction,\
    evaluate_on_samples,\
    KLController, RolloutBuffer, DictRolloutBuffer, MaskableDictRolloutBuffer,\
    TransitionInfo, TensorDict, RefPolicyOutput, ValueOutput, PolicyOutput
from benchmark.torch.RL4LMs.registry import DataPoolRegistry, MetricRegistry, RewardFunctionRegistry, \
    ModelRegistry, AlgorithmRegistry
from benchmark.torch.RL4LMs.env import TextGenEnv
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          AutoModelForSeq2SeqLM,
                          Trainer,
                          TrainingArguments,
                          DataCollatorForLanguageModeling,
                          DataCollatorForSeq2Seq)
from benchmark.torch.RL4LMs.env import LocalParallelVecEnv, make_vec_env
from transformers import PreTrainedTokenizer
from benchmark.torch.RL4LMs.summarization import RL4LMsSummaAgent
from benchmark.torch.RL4LMs.algorithms import RL4LMPPO
import torch
from parl.utils import logger

def build_tokenizer(tokenizer_config: Dict[str, Any]):
    logger.info(f"loading tokenizer of [{tokenizer_config['model_name']}] model")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_config["model_name"])
    if tokenizer.pad_token is None and tokenizer_config.get("pad_token_as_eos_token", True):
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = tokenizer_config.get(
        "padding_side", "left")
    tokenizer.truncation_side = tokenizer_config.get(
        "truncation_side", "left")
    return tokenizer


def build_reward_fn(reward_config: Dict[str, Any]):
    reward_fn = RewardFunctionRegistry.get(reward_config["id"],
                                           reward_config.get("args", {}))
    return reward_fn


def build_metrics(metric_configs: List[Dict[str, Any]]):
    metrics = [MetricRegistry.get(metric_config["id"], metric_config.get("args", {}))
               for metric_config in metric_configs]
    return metrics


def build_datapool(datapool_config: Dict[str, Any]):

    def _get_datapool_by_split(split: str):
        kwargs = datapool_config.get("args", {})
        kwargs["split"] = split
        dp_split = DataPoolRegistry.get(datapool_config["id"], kwargs)
        return dp_split

    train_datapool = _get_datapool_by_split("train")
    val_datapool = _get_datapool_by_split("val")
    test_datapool = _get_datapool_by_split("test")

    samples_by_split = {
        "train": [(sample, weight)
                  for sample, weight in train_datapool],
        "val": [sample for sample, _ in val_datapool],
        "test": [sample for sample, _ in test_datapool]
    }
    return samples_by_split


def build_env(env_config: Dict[str, Any],
              reward_fn: RewardFunction,
              tokenizer: AutoTokenizer,
              train_samples: List[Sample]):
    # vectoried env
    env_kwargs = {
        "reward_function": reward_fn,
        "tokenizer": tokenizer,
        "samples": train_samples,
    }
    env_kwargs = {**env_kwargs, **env_config.get("args", {})}
    envs = make_vec_env(TextGenEnv,
                       n_envs=env_config.get(
                           "n_envs", 1),
                       vec_env_cls=LocalParallelVecEnv,
                       env_kwargs=env_kwargs)
    return envs

def build_agent(alg_config: Dict[str, Any],
            env: LocalParallelVecEnv,
            model_state: Dict[str, Any] = None, # TODO: save model checkpoint
            device = None,
            alg_state: Dict[str, Any] = None    # TODO: save alg checkpoint
                ):
    model_config = alg_config["model"]
    model_cls = ModelRegistry.get(model_config["id"])
    alg_cls = AlgorithmRegistry.get(alg_config["id"])

    model_args = model_config["args"]
    model_args["state_dict"] = model_state

    rl4lms_model = model_cls(
        observation_space = env.observation_space,
        action_space= env.action_space,
        device=device,
        **model_args
    )

    rl4lm_alg_cls = alg_cls(
        model=rl4lms_model,
        device=device,
        **alg_config.get("args")
    )

    rl4lm_agent = RL4LMsSummaAgent(rl4lm_alg_cls, alg_config)
    return rl4lm_agent


def dict_to_tensor(obs, device):
    return {key: torch.as_tensor(_obs).to(device) for (key, _obs) in obs.items()}


def unpack_observations(obs_tensor, n_envs: int):
    """
    Unpacks vectorized dict observations into separate dict observations
    """
    unpacked_obs = []
    keys = obs_tensor.keys()
    for env_ix in range(n_envs):
        obs_dict = {}
        for key in keys:
            obs_dict[key] = obs_tensor[key][env_ix].reshape(1, -1).cpu()
        unpacked_obs.append(obs_dict)
    return unpacked_obs


class OnPolicyTrainer():
    """
    A generic trainer for training LMs with onpolicy algorithms from SB3
    """

    def __init__(self,
                 tokenizer_config: Dict[str, Any],
                 datapool_config: Dict[str, Any],
                 reward_config: Dict[str, Any],
                 env_config: Dict[str, Any],
                 on_policy_alg_config: Dict[str, Any],
                 train_eval_config: Dict[str, Any],
                 experiment_name: str = ''
                 ):
        self._tokenizer_config = tokenizer_config
        self._datapool_config = datapool_config
        self._reward_config = reward_config
        self._env_config = env_config
        self._on_policy_alg_config = on_policy_alg_config
        self._train_eval_config = train_eval_config
        self._experiment_name = experiment_name
        self._agent = None
        self._env = None
        self.num_timesteps = None
        self._kl_controller = None
        self.device = torch.device("cuda" if torch.cuda.
                                   is_available() else "cpu")
        self._norm_reward = False

        self._setup()

    def _setup(self):

        # load trainer state from available previous checkpoint if available
        # self.load_trainer_state(self._tracker)

        # build components
        self._tokenizer = build_tokenizer(self._tokenizer_config)
        self._reward_fn = build_reward_fn(self._reward_config)
        self._metrics = build_metrics(
            self._train_eval_config.get("metrics", []))
        self._samples_by_split = build_datapool(self._datapool_config)
        self._env = build_env(self._env_config, self._reward_fn,
                              self._tokenizer, self._samples_by_split["train"])


        self._agent = build_agent(self._on_policy_alg_config,
                                  self._env, device=self.device)

        self._rollout_buffer = MaskableDictRolloutBuffer(
            buffer_size=self._agent.alg.n_steps * self._env.num_envs,
            observation_space=self._agent.alg.model.observation_space,
            action_space=self._agent.alg.model.action_space,
            device=self.device,
            gamma=self._agent.alg.gamma,
            gae_lambda=self._agent.alg.gae_lambda,
            n_envs=1,
        )

        self._kl_controller = KLController(
            self._on_policy_alg_config["kl_div"]["coeff"],
            self._on_policy_alg_config["kl_div"].get("target_kl", None))

        # extract train params
        self._max_episode_length = self._env_config["args"]["max_episode_length"]
        self._max_prompt_length = self._env_config["args"]["max_prompt_length"]
        self._eval_batch_size = self._train_eval_config["eval_batch_size"]
        self._n_iters = int(self._train_eval_config["n_iters"])
        self._n_steps_per_iter = self._env.num_envs * self._agent.alg.n_steps
        self._num_timesteps = 0

        # gen kwargs for evaluation (if it is different from rollout gen kwargs)
        self._eval_gen_kwargs = self._train_eval_config.get(
            "generation_kwargs", None)

    def _evaluate_on_datapools(self, epoch: int,
                               splits: List[str] = ["val", "test"]):
        for split in splits:
            evaluate_on_samples(policy=self._agent.alg.model,
                                tokenizer=self._tokenizer,
                                samples=self._samples_by_split[split],
                                batch_size=self._eval_batch_size,
                                max_prompt_length=self._max_prompt_length,
                                metrics=self._metrics,
                                epoch=epoch,
                                split_name=split,
                                gen_kwargs=self._eval_gen_kwargs)

    def train_and_eval(self):
        # evaluate on val and test set before fine-tuning once
        # iter_start = self._trainer_state["current_iter"]
        iter_start = 0
        self._evaluate_on_datapools(epoch=iter_start)

        # train for given number of iters
        for epoch in range(iter_start, self._n_iters):
            print("========== BEGIN ==========")
            print(f"outer epoch: {epoch} / {self._n_iters - 1}")
            print("========== BEGIN ==========")
            outer_start_time = time.time()
            # current state
            # self._trainer_state["current_iter"] = epoch

            self._num_timesteps = 0

            while self._num_timesteps < self._n_steps_per_iter:
                self.collect_rollouts(self._env, self._rollout_buffer)
                # inner rollout and learn loop for on-policy algorithm
                # self._agent.learn(self._n_steps_per_iter)
                self._agent.learn(self._rollout_buffer)

            # save the policy checkpoint
            # if (epoch + 1) % self._train_eval_config.get("save_every", 20) == 0:
            #     self.save_trainer_state(
            #         self._tracker, self._alg.policy, self._trainer_state)

            # evaluate on val set in the given intervals
            if (epoch + 1) % self._train_eval_config["eval_every"] == 0:
                self._evaluate_on_datapools(epoch=epoch, splits=["val"])

            outer_end_time = time.time()
            print("========== END ==========")
            print(f"outer epoch: {epoch} / {self._n_iters - 1}")
            print(f"time used: {outer_end_time - outer_start_time} second(s), left time:"
                  f"  {1.0 * (outer_end_time - outer_start_time) * (self._n_iters - epoch - 1) / 60 / 60} hour(s)")
            print("========== END ==========")


        # finally evaluate on val and test samples
        self._evaluate_on_datapools(epoch=epoch)

        # # save model here - we save only the language model
        # if self._tracker is not None:
        #     self._tracker.save_auto_model(
        #         self._alg.policy.get_language_model())


    def get_policy_kwargs(
        self,
        obs: TensorDict,
        action: torch.tensor,
        past_state: Dict[str, torch.tensor],
        action_mask: torch.tensor,
    ):

        policy_kwargs = {
            "obs": obs,
            "actions": action,
            "past_model_kwargs": past_state,
        }
        if action_mask is not None:
            policy_kwargs["action_masks"] = action_mask
        return policy_kwargs

    def generate_batch(
        self,
        rollout_buffer,
        tokenizer: PreTrainedTokenizer,
        max_steps: int,
        rollout_info: Dict[str, Any],
    ):
        # if rollout buffer is already full, do not continue
        if rollout_buffer.full:
            return

        # start parallel episodes
        current_obs = self._env.reset()
        episode_starts = np.ones((self._env.num_envs,), dtype=bool)

        # generate text using the model
        obs_tensor =  dict_to_tensor(current_obs, self.device)
        generation_inputs = self._agent.alg.model.get_inputs_for_generation(obs_tensor)
        gen_output = self._agent.alg.model.generate(
            input_ids=generation_inputs.inputs,
            attention_mask=generation_inputs.attention_masks,
            tokenizer=tokenizer,
        )

        # process them one step at a time to collect rollout info
        episode_wise_transitions = [[] for _ in range(self._env.num_envs)]
        ep_terminated = np.zeros((self._env.num_envs,), dtype=bool)
        value_past_state = None
        ref_past_state = None
        policy_past_state = None
        masks = (
            gen_output.action_masks
            if gen_output.action_masks is not None
            else [None] * len(gen_output.step_wise_logprobs)
        )

        for actions_tensor, _, action_mask in zip(
            gen_output.step_wise_actions, gen_output.step_wise_logprobs, masks
        ):
            # if all episodes are done, just break and do not continue
            if np.all(ep_terminated):
                break

            # evaluate actions with actions from rollout
            with torch.no_grad():
                obs_tensor = dict_to_tensor(current_obs, self.device)

                # get log probs (TBD: generalize this a bit)
                policy_kwargs = self.get_policy_kwargs(
                    obs_tensor, actions_tensor, policy_past_state, action_mask
                )

                policy_outputs: PolicyOutput = self._agent.alg.model.forward_policy(
                    **policy_kwargs
                )
                raw_log_probs, log_probs, policy_past_state = (
                    policy_outputs.raw_log_probs,
                    policy_outputs.log_probs,
                    policy_outputs.past_model_kwargs,
                )

                # sanity check
                assert torch.all(
                    torch.isfinite(log_probs)
                ), "Infinite values in log probs"

                # sanity check
                assert torch.all(
                    torch.isfinite(raw_log_probs)
                ), "Infinite values in log probs"

                # get values
                value_outputs: ValueOutput = self._agent.alg.model.forward_value(
                    obs_tensor, value_past_state
                )
                values, value_past_state = (
                    value_outputs.values,
                    value_outputs.past_model_kwargs,
                )

                # get reference log probs
                ref_policy_outputs: RefPolicyOutput = (
                    self._agent.alg.model.get_log_probs_ref_model(
                        obs_tensor, actions_tensor, ref_past_state
                    )
                )
                ref_log_probs, ref_past_state = (
                    ref_policy_outputs.log_probs,
                    ref_policy_outputs.past_model_kwargs,
                )

                # sanity check
                assert torch.all(
                    torch.isfinite(ref_log_probs)
                ), "Infinite values in log probs"

                # compute KL rewards
                kl_div = raw_log_probs - ref_log_probs
                kl_rewards = -1 * self._kl_controller.kl_coeff * kl_div

            # step into env to get rewards
            actions = actions_tensor.cpu().numpy()
            new_obs, rewards, dones, infos = self._env.step(actions)

            self._num_timesteps += self._env.num_envs

            # compute total rewards
            total_rewards = rewards + kl_rewards.cpu().numpy()

            # unpack individual observations
            unpacked_obs = unpack_observations(obs_tensor, self._env.num_envs)

            # store episode wise transitions separately
            for env_ix in range(self._env.num_envs):
                # only if not terminated already
                if not ep_terminated[env_ix]:
                    transtion = TransitionInfo(
                        observation=unpacked_obs[env_ix],
                        action=actions[env_ix],
                        task_reward=rewards[env_ix],
                        total_reward=total_rewards[env_ix],
                        kl_div=kl_div.cpu().numpy()[env_ix],
                        episode_start=episode_starts[env_ix],
                        value=values[env_ix].cpu(),
                        log_prob=log_probs[env_ix].cpu(),
                        done=dones[env_ix],
                        ref_log_prob=ref_log_probs[env_ix].cpu(),
                        kl_reward=kl_rewards.cpu().numpy()[env_ix],
                        action_mask=action_mask[env_ix].cpu().numpy()
                        if action_mask is not None
                        else None,
                        info=infos[env_ix],
                    )

                    episode_wise_transitions[env_ix].append(transtion)

                # mark this episode to terminated if done occurs once
                if dones[env_ix]:
                    ep_terminated[env_ix] = True

            episode_starts = np.zeros((self._env.num_envs,), dtype=bool)
            current_obs = new_obs

        # now we flush all episode wise info to the 1-D buffer
        rollout_info = self._add_to_buffer(
            rollout_buffer, episode_wise_transitions, rollout_info
        )
        return rollout_info

    def _add_to_buffer(
        self, rollout_buffer, episode_wise_transitions, rollout_info
    ):
        # if the reward function is batchable, we override the rewards here
        # if isinstance(self.reward_fn, BatchedRewardFunction):
        #     compute_batched_rewards(episode_wise_transitions, self.reward_fn)

        advantages_computed = False
        for ep_ix, transitions in enumerate(episode_wise_transitions):
            ep_length = len(transitions)
            total_reward = 0.0
            total_kl_reward = 0.0
            for transition_ix, transition in enumerate(transitions):
                total_reward += transition.task_reward
                total_kl_reward += transition.kl_reward
                rollout_info["rollout_info/kl_div_mean"].append(transition.kl_div)
                rollout_info["rollout_info/log_prob"].append(transition.log_prob)
                rollout_info["rollout_info/ref_log_prob"].append(
                    transition.ref_log_prob
                )
                rollout_info["rollout_info/values"].append(transition.value.numpy())

                if not rollout_buffer.full:
                    rollout_buffer.add(
                        transition.observation,
                        transition.action,
                        transition.total_reward,
                        transition.episode_start,
                        transition.value,
                        transition.log_prob,
                        action_masks=transition.action_mask,
                    )

                # if the buffer is full, compute advantages
                if rollout_buffer.full and not advantages_computed:

                    # normalize the rewards
                    if self._norm_reward:
                        mean = rollout_buffer.rewards.mean()
                        std = rollout_buffer.rewards.std()
                        rollout_buffer.rewards = (rollout_buffer.rewards - mean) / (
                            std + 1e-8
                        )

                    # we fetch the last value for the last time step
                    # values come from the next transitions's values
                    next_values = (
                        transitions[transition_ix + 1].value
                        if (transition_ix + 1) < ep_length
                        else torch.tensor([0.0])
                    )

                    rollout_buffer.compute_returns_and_advantage(
                        last_values=next_values, dones=transition.done
                    )
                    advantages_computed = True

            rollout_info["rollout_info/ep_rew"].append(total_reward)
            rollout_info["rollout_info/ep_lens"].append(ep_length)
            rollout_info["rollout_info/ep_kl_rew"].append(total_kl_reward)
        return rollout_info

    def collect_rollouts(
        self,
        env,
        rollout_buffer: RolloutBuffer,
    ) -> bool:
        # max episode steps
        max_steps = env.get_attr("max_steps", [0])[0]

        # get tokenizer
        tokenizer = env.get_attr("tokenizer", [0])
        tokenizer = tokenizer[0]

        # Switch to eval mode
        self._agent.alg.model.set_training_mode(False)

        # reset rollout buffer and stats
        rollout_buffer.reset()

        # start the rollout process
        rollout_info = {
            "rollout_info/ep_rew": [],
            "rollout_info/kl_div_mean": [],
            "rollout_info/ep_lens": [],
            "rollout_info/ep_kl_rew": [],
            "rollout_info/log_prob": [],
            "rollout_info/ref_log_prob": [],
            "rollout_info/values": [],
        }
        while not rollout_buffer.full:
            # generate batch of rollouts
            rollout_info = self.generate_batch(
                rollout_buffer, tokenizer, max_steps, rollout_info
            )

        # aggregate rollout info
        aggregated_rollout_info = {}
        for key, values in rollout_info.items():
            aggregated_rollout_info[key] = np.mean(values).item()
            aggregated_rollout_info[f"{key}_std"] = np.std(values).item()
        aggregated_rollout_info[
            "rollout_info/kl_coeff"
        ] = self._kl_controller.kl_coeff

        # if self.tracker is not None:
        #     self.tracker.log_rollout_infos(aggregated_rollout_info)

        # adapt the KL coeff
        self._kl_controller.step(
            torch.tensor(aggregated_rollout_info["rollout_info/kl_div_mean"])
        )
        return True