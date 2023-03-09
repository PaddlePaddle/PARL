import torch
import numpy as np

from .kl_controller import KLController
from parl.utils import logger


def dict_to_tensor(obs, device):
    return {key: torch.as_tensor(_obs).to(device) for (key, _obs) in obs.items()}


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
    def __init__(self, kl_args, reviewer_group):
        self._kl_controller = KLController(kl_args["coeff"],
                                           kl_args["target_kl"])

    def _generate_batch(
            self,
            agent=None,
            reviewer_group=None,
            rollout_buffer=None,
            tokenizer=None,
            rollout_info=None,
            device=None
    ):
        # if rollout buffer is already full, do not continue
        if rollout_buffer.full:
            return

        # start parallel episodes
        current_obs = reviewer_group.ask()


        # generate text using the model
        obs_tensor = dict_to_tensor(current_obs, device)
        generation_inputs = agent.get_inputs_for_generation(obs_tensor)
        gen_output = agent.sample(
            input_ids=generation_inputs.inputs,
            attention_mask=generation_inputs.attention_masks,
            tokenizer=tokenizer,
        )

        episode_wise_transitions, num_timesteps = reviewer_group.feedback(current_obs=current_obs,
                                                                               gen_output=gen_output,
                                                                               kl_criterion=self._kl_controller,
                                                                               agent=agent,
                                                                               device=device)

        # now we flush all episode wise info to the 1-D buffer
        rollout_info = add_to_buffer(
            rollout_buffer, episode_wise_transitions, rollout_info
        )
        return rollout_info, num_timesteps


    def collect_rollouts(
            self,
            agent,
            reviewer_group,
            rollout_buffer,
            device
    ):
        # get tokenizer
        tokenizer = reviewer_group.tokenizer

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
                reviewer_group=reviewer_group,
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

        logger.info(f"Rollout Info: {aggregated_rollout_info}")


        # adapt the KL coeff
        self._kl_controller.step(
            torch.tensor(aggregated_rollout_info["rollout_info/kl_div_mean"])
        )
        return num_timesteps