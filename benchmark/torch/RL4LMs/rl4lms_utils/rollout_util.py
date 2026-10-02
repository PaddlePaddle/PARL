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

import torch
import numpy as np

from .kl_controller import KLController
from parl.utils import logger
from collections import OrderedDict
from .data_wrapper import TransitionInfo


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


class RolloutUtil(object):
    def __init__(self, kl_args):
        self._kl_controller = KLController(kl_args["coeff"], kl_args["target_kl"])

    def collect_rollouts(self, agent, instructor_group, rollout_buffer):
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

            # note: RL4LMs uses the same way (language model always does sample() to generate in summarization
            #       task) for collecting data and testing, so here agent uses predict() rather than sample()
            gen_output = agent.predict(dict_obs_tensor=current_obs, tokenizer=tokenizer)

            # get episode state, reward, dones, infos from instructors
            sentence_new_obs, sentence_rewards, sentence_dones, sentence_infos = instructor_group.feedback_sentense(
                gen_output=gen_output)

            # generate batch of rollouts and add to buffer
            episode_wise_transitions, run_timesteps = self._generate_transition(
                gen_sentence=gen_output,
                init_obs=current_obs,
                agent=agent,
                n_instructors=instructor_group.n_instructors,
                obs_space=instructor_group.observation_space,
                sentence_new_obs=sentence_new_obs,
                sentence_rewards=sentence_rewards,
                sentence_dones=sentence_dones,
                sentence_infos=sentence_infos,
            )
            num_timesteps += run_timesteps

            # now we flush all episode wise info to the 1-D buffer
            # log transition and add to buffer
            rollout_buffer.add_transitions(episode_wise_transitions, rollout_info)

        # aggregate rollout info
        aggregated_rollout_info = {}
        for key, values in rollout_info.items():
            aggregated_rollout_info[key] = np.mean(values).item()
            aggregated_rollout_info[f"{key}_std"] = np.std(values).item()
        aggregated_rollout_info["rollout_info/kl_coeff"] = self._kl_controller.kl_coeff

        logger.info(f"Rollout Info: {aggregated_rollout_info}")

        # adapt the KL coeff
        self._kl_controller.step(torch.tensor(aggregated_rollout_info["rollout_info/kl_div_mean"]))
        return num_timesteps

    def _generate_transition(self,
                             gen_sentence=None,
                             agent=None,
                             n_instructors=None,
                             obs_space=None,
                             sentence_new_obs=None,
                             sentence_rewards=None,
                             sentence_dones=None,
                             sentence_infos=None,
                             init_obs=None):
        current_obs = init_obs

        review_times = 0
        episode_starts = np.ones((n_instructors, ), dtype=bool)
        # process them one step at a time to collect rollout info
        episode_wise_transitions = [[] for _ in range(n_instructors)]
        ep_terminated = np.zeros((n_instructors, ), dtype=bool)

        for idx, actions_tensor in enumerate(gen_sentence.step_wise_actions):
            if np.all(ep_terminated):
                break

            # evaluate actions with actions from rollout
            with torch.no_grad():
                # prepare here for forward of value_model, policy_model and ref_model
                obs_tensor = agent.prepare_obs_input(current_obs)

                log_probs, _, _ = agent.policy(obs=obs_tensor, actions=actions_tensor)

                # sanity check
                assert torch.all(torch.isfinite(log_probs)), "Infinite values in log probs"

                # get values
                values, _ = agent.value(obs_tensor)

                # get reference log probs
                ref_log_probs, _, _ = agent.ref_policy(obs_tensor, actions_tensor)

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

            episode_starts = np.zeros((n_instructors, ), dtype=bool)
            current_obs = new_obs

        return episode_wise_transitions, review_times
