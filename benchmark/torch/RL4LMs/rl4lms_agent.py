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

import parl
import numpy as np
from gym import spaces
import torch
from parl.utils import logger


def explained_variance(y_pred, y_true):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


class RL4LMsAgent(parl.Agent):
    def __init__(
            self,
            algorithm,
            n_epochs,
            batch_size=64,
            norm_reward=False,
    ):
        super(RL4LMsAgent, self).__init__(algorithm)
        self.dataset = None
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self._norm_reward = norm_reward
        self._n_updates = 0
        self.device = self.alg.model.device

    def learn(self, rollout_buffer):
        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        approx_kl_divs = []

        loss = torch.tensor(0.0)

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            continue_training = True

            for batch_ix, rollout_data in enumerate(list(rollout_buffer.get(self.batch_size))):
                batch_action = rollout_data.actions
                if isinstance(self.alg.model.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    batch_action = rollout_data.actions.long().flatten()
                batch_obs = rollout_data.observations
                batch_adv = rollout_data.advantages
                batch_logprob = rollout_data.old_log_prob
                batch_return = rollout_data.returns
                batch_value = rollout_data.old_values

                continue_training, alg_learn_info = self.alg.learn(
                                    batch_obs=batch_obs,
                                    batch_action=batch_action,
                                    batch_value=batch_value,
                                    batch_return=batch_return,
                                    batch_logprob=batch_logprob,
                                    batch_adv=batch_adv)

                entropy_losses.append(alg_learn_info["entropy_losses"])
                pg_losses.append(alg_learn_info["pg_losses"])
                value_losses.append(alg_learn_info["value_losses"])
                clip_fractions.append(alg_learn_info["clip_fractions"])
                approx_kl_divs.append(alg_learn_info["approx_kl_divs"])
                if not continue_training:
                    break

            self._n_updates += 1 # according to stable-baseline3
            if not continue_training:
                print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_divs[-1]:.2f}")
                break

        # self._n_updates += self.n_epochs # change original RL4LMs code
        explained_var = explained_variance(rollout_buffer.values.flatten(), rollout_buffer.returns.flatten())

        # Logs
        train_info = {
            "train/entropy_loss": np.mean(entropy_losses),
            "train/policy_gradient_loss": np.mean(pg_losses),
            "train/value_loss": np.mean(value_losses),
            "train/approx_kl": np.mean(approx_kl_divs),
            "train/clip_fraction": np.mean(clip_fractions),
            "train/loss": loss.item(),
            "train/explained_variance": explained_var
        }

        if hasattr(self.alg.model, "log_std"):
            # self.logger.record(
            #     "train/std", torch.exp(self.policy.log_std).mean().item())
            train_info["train/std"] = torch.exp(self.alg.model.log_std).mean().item()

        # self.logger.record("train/n_updates",
        #                    self._n_updates, exclude="tensorboard")
        # self.logger.record("train/clip_range", clip_range)
        train_info["train/n_updates"] = self._n_updates
        train_info["train/clip_param"] = self.alg.clip_param

        logger.info(train_info)

        ppo_train_info = {
            "ppo/entropy_loss": np.mean(entropy_losses).item(),
            "ppo/policy_gradient_loss": np.mean(pg_losses).item(),
            "ppo/value_loss": np.mean(value_losses).item(),
            "ppo/approx_kl": np.mean(approx_kl_divs).item(),
        }

        logger.info(ppo_train_info)

    def get_inputs_for_generation(self, dict_obs_tensor):
        obs_tensor = self.prepare_obs_input(dict_obs_tensor)
        return self.alg.model.get_inputs_for_generation(obs_tensor)

    def prepare_obs_input(self, obs):
        return {key: torch.as_tensor(_obs).to(self.device) for (key, _obs) in obs.items()}

    def value(
            self,
            obs,
    ):
        return self.alg.value(obs)

    # note: RL4LMs uses the same way (language model always does sample() to generate in summarization task) for
    #       collecting data and testing, so here use policy() instead of sample() and only need to return info
    #       like log_prob and gen_kwargs without action
    def policy(
            self,
            obs,
            actions,
    ):
        return self.alg.policy(
            obs=obs,
            actions=actions,
        )

    def get_log_probs_ref_model(
            self,
            obs,
            action,
    ):
        return self.alg.get_log_probs_ref_model(obs, action)

    def predict(
            self,
            tokenizer,
            texts=None,
            max_prompt_length=None,
            input_ids=None,
            attention_mask=None,
            gen_kwargs=None,
    ):
        return self.alg.predict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
            texts=texts,
            max_prompt_length=max_prompt_length,
            gen_kwargs=gen_kwargs)

    def eval_mode(self):
        self.alg.eval_mode()
