import torch
import numpy as np
from .data_wrapper import TransitionInfo
from .kl_controller import KLController


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
        self._kl_controller = KLController(kl_args["coeff"],
                                           kl_args["target_kl"])

    def _generate_batch(
            self,
            agent=None,
            env=None,
            rollout_buffer=None,
            tokenizer=None,
            rollout_info=None,
            device=None
    ):
        num_timesteps = 0
        # if rollout buffer is already full, do not continue
        if rollout_buffer.full:
            return

        # start parallel episodes
        current_obs = env.reset()
        episode_starts = np.ones((env.num_envs,), dtype=bool)

        # generate text using the model
        obs_tensor = dict_to_tensor(current_obs, device)
        generation_inputs = agent.get_inputs_for_generation(obs_tensor)
        gen_output = agent.sample(
            input_ids=generation_inputs.inputs,
            attention_mask=generation_inputs.attention_masks,
            tokenizer=tokenizer,
        )

        # process them one step at a time to collect rollout info
        episode_wise_transitions = [[] for _ in range(env.num_envs)]
        ep_terminated = np.zeros((env.num_envs,), dtype=bool)

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

                policy_outputs = agent.forward_policy(
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
                value_outputs = agent.forward_value(
                    obs_tensor
                )
                values, value_past_state = (
                    value_outputs.values,
                    value_outputs.past_model_kwargs,
                )

                # get reference log probs
                ref_policy_outputs = (
                    agent.get_log_probs_ref_model(
                        obs_tensor, actions_tensor
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
            new_obs, rewards, dones, infos = env.step(actions)

            num_timesteps += env.num_envs

            # compute total rewards
            total_rewards = rewards + kl_rewards.cpu().numpy()

            # unpack individual observations
            unpacked_obs = unpack_observations(obs_tensor, env.num_envs)

            # store episode wise transitions separately
            for env_ix in range(env.num_envs):
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

            episode_starts = np.zeros((env.num_envs,), dtype=bool)
            current_obs = new_obs

        # now we flush all episode wise info to the 1-D buffer
        rollout_info = add_to_buffer(
            rollout_buffer, episode_wise_transitions, rollout_info
        )
        return rollout_info, num_timesteps


    def collect_rollouts(
            self,
            agent,
            env,
            rollout_buffer,
            device
    ):
        # get tokenizer
        tokenizer = env.tokenizer

        # Switch to eval mode
        # self._agent.alg.model.set_training_mode(False)
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
            # generate batch of rollouts
            rollout_info, run_timesteps = self._generate_batch(
                agent=agent,
                env=env,
                rollout_buffer=rollout_buffer,
                tokenizer=tokenizer,
                rollout_info=rollout_info,
                device=device
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

        # adapt the KL coeff
        self._kl_controller.step(
            torch.tensor(aggregated_rollout_info["rollout_info/kl_div_mean"])
        )
        return num_timesteps