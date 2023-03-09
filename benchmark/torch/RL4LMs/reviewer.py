import gym
from collections import OrderedDict
import torch
from rl4lms_utils import TransitionInfo, Sample, Observation
from gym import Env, spaces
from gym.spaces.dict import Dict as DictSpace
from gym.spaces.discrete import Discrete
import parl
from collections import deque
import numpy as np

def _flatten_obs(obs, space):
    assert isinstance(obs, (list, tuple)), "expected list or tuple of observations per environment"
    assert len(obs) > 0, "need observations from at least one environment"

    if isinstance(space, gym.spaces.Dict):
        assert isinstance(space.spaces, OrderedDict), "Dict space must have ordered subspaces"
        assert isinstance(obs[0], dict), "non-dict observation for environment with Dict observation space"
        return OrderedDict([(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()])
    elif isinstance(space, gym.spaces.Tuple):
        assert isinstance(obs[0], tuple), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple(np.stack([o[i] for o in obs]) for i in range(obs_len))
    else:
        return np.stack(obs)

def dict_to_tensor(obs, device):
    return {key: torch.as_tensor(_obs).to(device) for (key, _obs) in obs.items()}

def unpack_observations(obs_tensor, n_envs):
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


@parl.remote_class(wait=False)
class Reviewer:
    def __init__(
        self,
        tokenizer,
        reward_function,
        samples,
        max_episode_length = 512,
        max_prompt_length = None,
        terminate_on_eos = False,
        context_start_token = None,
        prompt_truncation_side = "left",
    ):

        """
        A generic RL environment to generate textual sequences.
        For eg: text generation, summarization, machine translation, text simplification
        Args:
            tokenizer (AutoTokenizer): pre-trained tokenizer
            reward_function (RewardFunction): reward functiom
            samples (Tuple[List[Sample], float]): list of samples
            max_episode_length (int, optional): Max steps to the model Defaults to 512.
            max_prompt_length (Optional[int], optional): maximum prompt length. Defaults to None.
            terminate_on_eos (bool, optional): whether to terminate on EOS. Defaults to False.
            context_start_token (bool, optional): start token for the context (For Encoder-Decoder models! )
            prompt_truncation_side (str): truncation side for prompt text (Defaults to "left")
        """
        self.tokenizer = tokenizer
        self.reward_function = reward_function
        self.max_steps = max_episode_length
        self._max_text_length = (
            max_prompt_length if max_prompt_length else tokenizer.model_max_length
        )
        self._terminate_on_eos = terminate_on_eos
        self._context_start_token = context_start_token
        self._prompt_truncation_side = prompt_truncation_side
        super().__init__()

        # set the observation and action space here
        self._vocab_size = tokenizer.vocab_size
        self.observation_space = DictSpace(
            {
                # we have to provide fixed sized inputs (padded) because sb3 support for DictObsersevation is limited
                # while creating rollout buffers, observations are concatenated for each key
                "prompt_or_input_encoded_pt": spaces.Box(
                    low=0, high=self._vocab_size, shape=(self._max_text_length,)
                ),
                "prompt_or_input_attention_mask_pt": spaces.Box(
                    low=0, high=1, shape=(self._max_text_length,)
                ),
                "context_encoded_pt": spaces.Box(
                    low=0, high=self._vocab_size, shape=(self.max_steps,)
                ),
                "context_attention_mask_pt": spaces.Box(
                    low=0, high=1, shape=(self.max_steps,)
                ),
                "input_encoded_pt": spaces.Box(
                    low=0,
                    high=self._vocab_size,
                    shape=(self._max_text_length + self.max_steps,),
                ),
                "input_attention_mask_pt": spaces.Box(
                    low=0, high=1, shape=(self._max_text_length + self.max_steps,)
                ),
            }
        )
        self.action_space = Discrete(n=self._vocab_size)
        # see https://github.com/huggingface/transformers/issues/4875 : rounding up to nearest power of 2 for better GPU efficiency
        if 'mt5' in self.tokenizer.name_or_path:
            n = 250112
            self.action_space = Discrete(n=n)
        elif 't5' in self.tokenizer.name_or_path:
            n = 32128
            self.action_space = Discrete(n=n)
        self.samples_for_replaying = deque()
        for sample, weight in samples:
            self.samples_for_replaying.append(sample)

        # check the tokenizer and add padding tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # TBD: configure this
        self.tokenizer.truncation_side = "left"  # TBD: configure this

        # init tracking variables
        self.__current_sample = None
        self.__current_obs = None
        self.__time_step = None

    def get_new_obs_and_feedback_one_step(self, action):
        self.__time_step += 1

        # previous obs
        previous_obs = self.__current_obs

        # just update the context tensor and gets the new observation
        self.__current_obs = self.__current_obs.update(action, self.tokenizer)

        # decide if the episode is finished or not
        done = (action == self.tokenizer.eos_token_id and self._terminate_on_eos) or (
            self.__time_step == self.max_steps
        )

        # compute reward
        reward = self.reward_function(
                previous_obs,
                action,
                self.__current_obs,
                done,
                self.__current_obs.meta_info,
            )

        # populate additional info
        info = {
            "output": self.__current_obs.context_text,
            "action_history": self.__current_obs.action_history,
            "reference_text": self.__current_obs.target_or_reference_texts,
            "prompt_text": self.__current_obs.prompt_or_input_text,
            "prev_output": previous_obs.context_text,
            "meta_info": previous_obs.meta_info,
        }

        if done:
            # save final observation where user can get it, then reset
            info["terminal_observation"] = self.__current_obs.to_dict()
            observation = self.ask()
            return (observation, reward, done, info)
        else:
            return (self.__current_obs.to_dict(), reward, done, info)

    def ask(self, sample = None):
        """
        Resets the environment and starts a new episode
        """
        # gets a new sample if not provided
        if sample is None:
            sample = np.random.choice(a=self.samples_for_replaying, size=min(len(self.samples_for_replaying), 1))[0]
        self.__current_sample = sample

        # init the observation
        self.__current_obs = Observation.init_from_sample(
            sample,
            self.tokenizer,
            self._max_text_length,
            self.max_steps,
            self._prompt_truncation_side,
            self._context_start_token,
            sample.meta_data,
        )

        # start the time step counter
        self.__time_step = 0

        dict_observation = self.__current_obs.to_dict()
        return dict_observation

    def get_obs_and_action_space(self):
        return (self.observation_space, self.action_space)


class ReviewerGroup:
    def __init__(self,
                reviewer_config=None,
                reward_fn=None,
                tokenizer=None,
                question_samples=None,
                seed = None,
                start_index = 0,
                ):
        self.n_reviewers = reviewer_config["n_reviewers"]
        reviewer_kwargs = {
            "reward_function": reward_fn,
            "tokenizer": tokenizer,
            "samples": question_samples,
        }
        reviewer_kwargs = {**reviewer_kwargs, **reviewer_config.get("args", {})}
        self.tokenizer = tokenizer
        self._remote_reviewers = self._create_reviewers(reviewer_kwargs, reviewer_config["parl_master_address"])
        tem_future_object_ids = self._remote_reviewers[0].get_obs_and_action_space()
        self.observation_space, self.action_space = tem_future_object_ids.get()
        # self.observation_space, self.action_space = tem_future_object_ids

    def ask(self):
        future_object_ids = [
            remote_reviewer.ask() for remote_reviewer in self._remote_reviewers
        ]
        sample_questions = [
            future_object.get() for future_object in future_object_ids
        ]
        # sample_questions = future_object_ids
        return _flatten_obs(sample_questions, self.observation_space)

    def feedback(self, current_obs, gen_output, kl_criterion, agent, device):
        review_times = 0
        episode_starts = np.ones((self.n_reviewers,), dtype=bool)
        # process them one step at a time to collect rollout info
        episode_wise_transitions = [[] for _ in range(self.n_reviewers)]
        ep_terminated = np.zeros((self.n_reviewers,), dtype=bool)

        for actions_tensor, _ in zip(
                gen_output.step_wise_actions, gen_output.step_wise_logprobs
        ):
            # if all episodes are done, just break and do not continue
            if np.all(ep_terminated):
                break

            # evaluate actions with actions from rollout
            with torch.no_grad():
                obs_tensor = dict_to_tensor(current_obs, device)

                # get log probs (TBD: generalize this a bit)
                policy_kwargs = {
                    "obs": obs_tensor,
                    "actions": actions_tensor,
                }

                _, log_probs, _, _ = agent.forward_policy(**policy_kwargs)

                # sanity check
                assert torch.all(torch.isfinite(log_probs)), "Infinite values in log probs"

                # get values
                values, _ = agent.forward_value(obs_tensor)

                # get reference log probs
                ref_log_probs, _ = agent.get_log_probs_ref_model(obs_tensor, actions_tensor)

                # sanity check
                assert torch.all(torch.isfinite(ref_log_probs)), "Infinite values in log probs"

                # compute KL rewards
                kl_div = log_probs - ref_log_probs
                kl_rewards = -1 * kl_criterion.kl_coeff * kl_div

            # step into env to get rewards
            actions = actions_tensor.cpu().numpy()
            new_obs, rewards, dones, infos = self._feedback_one_step(actions)

            review_times += self.n_reviewers

            # compute total rewards
            total_rewards = rewards + kl_rewards.cpu().numpy()

            # unpack individual observations
            unpacked_obs = unpack_observations(obs_tensor, self.n_reviewers)

            # store episode wise transitions separately
            for env_ix in range(self.n_reviewers):
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
                        info=infos[env_ix],
                    )

                    episode_wise_transitions[env_ix].append(transtion)

                # mark this episode to terminated if done occurs once
                if dones[env_ix]:
                    ep_terminated[env_ix] = True

            episode_starts = np.zeros((self.n_reviewers,), dtype=bool)
            current_obs = new_obs
        return episode_wise_transitions, review_times

    def _feedback_one_step(self, actions):
        future_object_ids = [
            self._remote_reviewers[i].get_new_obs_and_feedback_one_step(
                actions[i]) for i in range(self.n_reviewers)
        ]
        feedback_res = [
            future_object.get() for future_object in future_object_ids
        ]
        # feedback_res = future_object_ids
        obs, rews, dones, infos = zip(*feedback_res)
        return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos


    def _create_reviewers(self, reviewer_kwargs, parl_port=None):
        parl.connect(parl_port, distributed_files=["./rl4lms_utils/*.py", "./*.py"])
        return [Reviewer(**reviewer_kwargs) for _ in range(self.n_reviewers)]




