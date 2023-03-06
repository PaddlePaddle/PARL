import parl
from benchmark.torch.RL4LMs.utils import Schedule
from typing import Union, Optional, Dict, Any
import torch
from gym import spaces
from benchmark.torch.RL4LMs.utils import EvaluateActionsOutput
from torch.nn import functional as F


from  parl.algorithms.torch import PPO

class RL4LMPPO(parl.Algorithm):
    def __init__(self,
                 model: parl.Model,
                 learning_rate: Union[float, Schedule] = 3e-4,
                 n_steps: int = 2048,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_range: Union[float, Schedule] = 0.2,
                 normalize_advantage: bool = True,
                 ent_coef: float = 0.0,
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 target_kl: Optional[float] = None,
                 seed: Optional[int] = None,
                 device: Union[torch.device, str] = "auto",
                 _init_setup_model: bool = True,
                 ):
        super(RL4LMPPO, self).__init__(model=model)
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
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

    def learn(self, rollout_buffer, log_info):
        entropy_losses = log_info["entropy_losses"]
        pg_losses = log_info["pg_losses"]
        value_losses = log_info["value_losses"]
        clip_fractions = log_info["clip_fractions"]
        approx_kl_divs = log_info["approx_kl_divs"]
        continue_training = True
        # Do a complete pass on the rollout buffer
        for batch_ix, rollout_data in enumerate(list(rollout_buffer.get(self.batch_size))):
            # self.verify_rollout_data(rollout_data)

            actions = rollout_data.actions
            if isinstance(self.model.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = rollout_data.actions.long().flatten()


            evaluation_output: EvaluateActionsOutput = self.model.evaluate_actions(
                rollout_data.observations, actions)
            values, log_prob, entropy = evaluation_output.values, evaluation_output.log_prob, evaluation_output.entropy
            values = values.flatten()
            # Normalize advantage
            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()
                              ) / (advantages.std() + 1e-8)

            # ratio between old and new policy, should be one at the first iteration
            ratio = torch.exp(log_prob - rollout_data.old_log_prob)

            # clipped surrogate loss
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * \
                            torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

            # Logging
            pg_losses.append(policy_loss.item())
            clip_fraction = torch.mean(
                (torch.abs(ratio - 1) > self.clip_range).float()).item()
            clip_fractions.append(clip_fraction)

            # No clipping
            values_pred = values

            # Value loss using the TD(gae_lambda) target
            value_loss = F.mse_loss(rollout_data.returns, values_pred)
            value_losses.append(value_loss.item())

            # Entropy loss favor exploration
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -torch.mean(-log_prob)
            else:
                entropy_loss = -torch.mean(entropy)

            entropy_losses.append(entropy_loss.item())

            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

            # Calculate approximate form of reverse KL Divergence for early stopping
            # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
            # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
            # and Schulman blog: http://joschu.net/blog/kl-approx.html
            with torch.no_grad():
                log_ratio = log_prob - rollout_data.old_log_prob
                approx_kl_div = torch.mean(
                    (torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                approx_kl_divs.append(approx_kl_div)

            if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                continue_training = False
                break

            # Optimization step
            self.model.optimizer.zero_grad()
            loss.backward()
            # Clip grad norm
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm)
            self.model.optimizer.step()

        return continue_training, loss


    def sample(self, obs):
        pass

    def predict(self, obs):
        pass

    def value(self, obs):
        pass

    def forward_value(
        self,
        obs,
        past_model_kwargs = None,
    ):
        return self.model.forward_value(obs, past_model_kwargs)

    def forward_policy(
        self,
        obs,
        actions: torch.tensor,
        past_model_kwargs = None,
    ):
        return self.model.forward_policy(
            obs = obs,
            actions = actions,
            past_model_kwargs = past_model_kwargs,
        )


    def get_log_probs_ref_model(
        self,
        obs,
        action,
        model_kwarpast_model_kwargsgs = None,
    ):
        return self.model.get_log_probs_ref_model(obs, action, model_kwarpast_model_kwargsgs)

    def generate(
        self,
        tokenizer,
        texts = None,
        max_prompt_length = None,
        input_ids = None,
        attention_mask = None,
        gen_kwargs = None,
    ):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
            texts = texts,
            max_prompt_length = max_prompt_length,
            gen_kwargs = gen_kwargs
        )

    def eval_mode(self):
        self.model.eval()