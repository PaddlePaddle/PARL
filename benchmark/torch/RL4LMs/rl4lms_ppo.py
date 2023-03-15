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
import torch
from torch.nn import functional as F


class RL4LMsPPO(parl.Algorithm):
    def __init__(
            self,
            model,
            learning_rate=3e-4,
            n_steps=2048,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            normalize_advantage=True,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=None,
            seed=None,
            device="auto",
            use_clipped_value_loss=False,
    ):
        super(RL4LMsPPO, self).__init__(model=model)
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.normalize_advantage = normalize_advantage
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.seed = seed
        self.device = device
        self.use_clipped_value_loss = use_clipped_value_loss
        for param_group in self.model.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

    def learn(self,
              batch_obs,
              batch_action,
              batch_logprob,
              batch_return,
              batch_adv):
        # Do a complete pass on the rollout batch
        continue_training = True
        learn_info = {"entropy_losses": None,
               "pg_losses": None,
               "value_losses": None,
               "clip_fractions": None,
               "approx_kl_divs": None,
                      "loss":None}

        values, action_log_probs, entropy = self.model.evaluate_actions(batch_obs, batch_action)
        values = values.flatten()

        # Normalize advantage
        if self.normalize_advantage:
            batch_adv = (batch_adv - batch_adv.mean()) / (batch_adv.std() + 1e-8)

        # ratio between old and new policy, should be one at the first iteration
        ratio = torch.exp(action_log_probs - batch_logprob)

        # clipped surrogate loss
        surr1 = ratio * batch_adv
        surr2 = torch.clamp(ratio, 1 - self.clip_range,
                             1 + self.clip_range) * batch_adv

        policy_loss = -torch.min(surr1, surr2).mean()

        # Logging
        learn_info["pg_losses"] = policy_loss.item()
        clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_range).float()).item()
        learn_info["clip_fractions"] = clip_fraction

        # No clipping
        values_pred = values

        # Value loss using the TD(gae_lambda) target
        value_loss = F.mse_loss(batch_return, values_pred)
        learn_info["value_losses"] = value_loss.item()

        # Entropy loss favor exploration
        if entropy is None:
            # Approximate entropy when no analytical form
            entropy_loss = -torch.mean(-action_log_probs)
        else:
            entropy_loss = -torch.mean(entropy)

        learn_info["entropy_losses"] = entropy_loss.item()

        loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

        # Calculate approximate form of reverse KL Divergence for early stopping
        # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
        # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
        # and Schulman blog: http://joschu.net/blog/kl-approx.html
        with torch.no_grad():
            log_ratio = action_log_probs - batch_logprob
            approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
            learn_info["approx_kl_divs"] = approx_kl_div

        learn_info["loss"] = loss

        if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
            continue_training = False
            return continue_training, learn_info

        # Optimization step
        self.model.optimizer.zero_grad()
        loss.backward()
        # Clip grad norm
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.model.optimizer.step()

        return continue_training, learn_info

    def value(
            self,
            obs,
    ):
        return self.model.value(obs)

    # note: RL4LMs uses the same way (language model always does sample() to generate in summarization
    #       task) for collecting data and testing, so here policy() only needs to return info
    #       like log_prob and gen_kwargs without action
    def policy(
            self,
            obs,
            actions,
    ):
        return self.model.policy(
            obs=obs,
            actions=actions,
        )

    def get_log_probs_ref_model(
            self,
            obs,
            action,
    ):
        return self.model.get_log_probs_ref_model(obs, action)

    def predict(
            self,
            tokenizer,
            texts=None,
            max_prompt_length=None,
            input_ids=None,
            attention_mask=None,
            gen_kwargs=None,
    ):
        return self.model.predict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
            texts=texts,
            max_prompt_length=max_prompt_length,
            gen_kwargs=gen_kwargs)

    def eval_mode(self):
        self.model.eval()
