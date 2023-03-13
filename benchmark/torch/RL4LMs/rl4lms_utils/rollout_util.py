import torch
import numpy as np

from .kl_controller import KLController
from parl.utils import logger
from collections import OrderedDict
from .data_wrapper import TransitionInfo


def dict_to_tensor(obs, device):
    return {key: torch.as_tensor(_obs).to(device) for (key, _obs) in obs.items()}

def get_one_token_obs(obs, idx, space):
    return OrderedDict([(k, obs[k][:, idx, :]) for k in space.spaces.keys()])

def unpack_observations(obs_tensor, n_instructors):
    """
    Unpacks vectorized dict observations into separate dict observations
    """
    unpacked_obs = []
    keys = obs_tensor.keys()
    for instructor_ix in range(n_instructors):
        obs_dict = {}
        for key in keys:
            obs_dict[key] = obs_tensor[key][instructor_ix].reshape(1, -1).cpu()
        unpacked_obs.append(obs_dict)
    return unpacked_obs


def add_to_buffer(
        rollout_buffer, episode_wise_transitions, rollout_info
):
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
                )

            # if the buffer is full, compute advantages
            if rollout_buffer.full and not advantages_computed:
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


class RolloutUtil:
    def __init__(self, kl_args):
        self._kl_controller = KLController(kl_args["coeff"], kl_args["target_kl"])

    def collect_rollouts(
            self,
            agent,
            instructor_group,
            rollout_buffer,
            device
    ):
        # get tokenizer
        tokenizer = instructor_group.tokenizer

        # Switch to eval mode both training and testing
        agent.eval_mode()

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
        num_timesteps = 0
        while not rollout_buffer.full:
            # start parallel episodes
            current_obs = instructor_group.ask()

            # generate sentences using the model
            obs_tensor = dict_to_tensor(current_obs, device)
            generation_inputs = agent.get_inputs_for_generation(obs_tensor)
            gen_output = agent.sample(
                input_ids=generation_inputs.inputs,
                attention_mask=generation_inputs.attention_masks,
                tokenizer=tokenizer)

            # get episode state, reward, dones, infos from instructors
            sentence_new_obs, sentence_rewards, sentence_dones, sentence_infos = instructor_group.feedback_sentense(
                gen_output=gen_output)

            # generate batch of rollouts and add to buffer
            rollout_info, run_timesteps = self._generate_transition_and_add_to_buffer(
                gen_sentence=gen_output,
                init_obs=current_obs,
                agent=agent,
                n_instructors=instructor_group.n_instructors,
                obs_space=instructor_group.observation_space,
                sentence_new_obs=sentence_new_obs,
                sentence_rewards=sentence_rewards,
                sentence_dones=sentence_dones,
                sentence_infos=sentence_infos,
                rollout_buffer=rollout_buffer,
                rollout_info=rollout_info,
                device=device,
            )
            num_timesteps += run_timesteps

        # aggregate rollout info
        aggregated_rollout_info = {}
        for key, values in rollout_info.items():
            aggregated_rollout_info[key] = np.mean(values).item()
            aggregated_rollout_info[f"{key}_std"] = np.std(values).item()
        aggregated_rollout_info[
            "rollout_info/kl_coeff"
        ] = self._kl_controller.kl_coeff

        logger.info(f"Rollout Info: {aggregated_rollout_info}")

        # adapt the KL coeff
        self._kl_controller.step(
            torch.tensor(aggregated_rollout_info["rollout_info/kl_div_mean"])
        )
        return num_timesteps

    def _generate_transition_and_add_to_buffer(
            self,
            gen_sentence=None,
            agent=None,
            n_instructors=None,
            obs_space=None,
            rollout_buffer=None,
            rollout_info=None,
            device=None,
            sentence_new_obs=None,
            sentence_rewards=None,
            sentence_dones=None,
            sentence_infos=None,
            init_obs=None
    ):
        current_obs = init_obs

        review_times = 0
        episode_starts = np.ones((n_instructors,), dtype=bool)
        # process them one step at a time to collect rollout info
        episode_wise_transitions = [[] for _ in range(n_instructors)]
        ep_terminated = np.zeros((n_instructors,), dtype=bool)


        for idx, actions_tensor in enumerate(gen_sentence.step_wise_actions):
            if np.all(ep_terminated):
                break

            # evaluate actions with actions from rollout
            with torch.no_grad():
                obs_tensor = dict_to_tensor(current_obs, device)

                _, log_probs, _, _ = agent.forward_policy(obs=obs_tensor, actions=actions_tensor)

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
                kl_rewards = -1 * self._kl_controller.kl_coeff * kl_div

            actions = actions_tensor.cpu().numpy()
            rewards = sentence_rewards[:, idx]
            dones = sentence_dones[:, idx]
            new_obs = get_one_token_obs(sentence_new_obs, idx, obs_space)
            infos = sentence_infos[:, idx]

            review_times += n_instructors

            # compute total rewards
            total_rewards = rewards + kl_rewards.cpu().numpy()

            # unpack individual observations
            unpacked_obs = unpack_observations(obs_tensor, n_instructors)

            # store episode wise transitions separately
            for instructor_ix in range(n_instructors):
                # only if not terminated already
                if not ep_terminated[instructor_ix]:
                    transtion = TransitionInfo(
                        observation=unpacked_obs[instructor_ix],
                        action=actions[instructor_ix],
                        task_reward=rewards[instructor_ix],
                        total_reward=total_rewards[instructor_ix],
                        kl_div=kl_div.cpu().numpy()[instructor_ix],
                        episode_start=episode_starts[instructor_ix],
                        value=values[instructor_ix].cpu(),
                        log_prob=log_probs[instructor_ix].cpu(),
                        done=dones[instructor_ix],
                        ref_log_prob=ref_log_probs[instructor_ix].cpu(),
                        kl_reward=kl_rewards.cpu().numpy()[instructor_ix],
                        info=infos[instructor_ix],
                    )

                    episode_wise_transitions[instructor_ix].append(transtion)

                # mark this episode to terminated if done occurs once
                if dones[instructor_ix]:
                    ep_terminated[instructor_ix] = True

            episode_starts = np.zeros((n_instructors,), dtype=bool)
            current_obs = new_obs

        # now we flush all episode wise info to the 1-D buffer
        rollout_info = add_to_buffer(
            rollout_buffer, episode_wise_transitions, rollout_info
        )
        return rollout_info, review_times
