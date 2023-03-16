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
from parl.utils.utils import check_model_method


class RL4LMsPPO(parl.Algorithm):
    def __init__(
            self,
            model,
            clip_param=0.2,
            value_loss_coef=0.5,
            entropy_coef=0.0,
            initial_lr=3e-4,
            max_grad_norm=0.5,
            use_clipped_value_loss=False,
            norm_adv=True,
            target_kl=None,
            seed=None,
    ):
        # check model method
        check_model_method(model, 'value', self.__class__.__name__)
        check_model_method(model, 'policy', self.__class__.__name__)

        assert isinstance(clip_param, float)
        assert isinstance(value_loss_coef, float)
        assert isinstance(entropy_coef, float)
        assert isinstance(initial_lr, float)
        assert isinstance(max_grad_norm, float)
        assert isinstance(use_clipped_value_loss, bool)
        assert isinstance(norm_adv, bool)

        super(RL4LMsPPO, self).__init__(model=model)
        self.initial_lr = initial_lr
        self.clip_param = clip_param
        self.norm_adv = norm_adv
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.seed = seed
        self.use_clipped_value_loss = use_clipped_value_loss

        for param_group in self.model.optimizer.param_groups:
            param_group["lr"] = self.initial_lr
        self.optimizer = self.model.optimizer

    def learn(self, batch_obs, batch_action, batch_value, batch_return, batch_logprob, batch_adv, lr=None):
        # Do a complete pass on the rollout batch
        continue_training = True
        learn_info = {
            "entropy_losses": None,
            "pg_losses": None,
            "value_losses": None,
            "clip_fractions": None,
            "approx_kl_divs": None,
            "loss": None
        }

        values, _ = self.model.value(batch_obs)
        action_log_probs, entropy, _ = self.model.policy(batch_obs, batch_action)
        values = values.flatten()
        entropy_loss = torch.mean(entropy)
        learn_info["entropy_losses"] = entropy_loss.item()

        # Normalize advantage
        if self.norm_adv:
            batch_adv = (batch_adv - batch_adv.mean()) / (batch_adv.std() + 1e-8)

        # ratio between old and new policy, should be one at the first iteration
        ratio = torch.exp(action_log_probs - batch_logprob)

        # clipped surrogate loss
        surr1 = ratio * batch_adv
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_adv

        action_loss = -torch.min(surr1, surr2).mean()

        # Logging
        learn_info["pg_losses"] = action_loss.item()
        clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_param).float()).item()
        learn_info["clip_fractions"] = clip_fraction

        # clipping
        # values_pred = values
        if self.use_clipped_value_loss:
            value_pred_clipped = batch_value + torch.clamp(
                values - batch_value,
                -self.clip_param,
                self.clip_param,
            )
            value_losses = (values - batch_return).pow(2)
            value_losses_clipped = (value_pred_clipped - batch_return).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = 0.5 * (batch_return - values).pow(2).mean()

        # Value loss using the TD(gae_lambda) target
        # value_loss = F.mse_loss(batch_return, values_pred)
        learn_info["value_losses"] = value_loss.item()

        loss = value_loss * self.value_loss_coef + action_loss - self.entropy_coef * entropy_loss

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

        if lr:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        # Clip grad norm
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return continue_training, learn_info

    def value(self, obs):
        return self.model.value(obs)

    # note: RL4LMs uses the same way (language model always does sample() to generate in summarization
    #       task) for collecting data and testing, so here policy() only needs to return info
    #       like log_prob and gen_kwargs without action
    def policy(self, obs, actions):
        return self.model.policy(
            obs=obs,
            actions=actions,
        )

    def get_log_probs_ref_model(self, obs, action):
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
