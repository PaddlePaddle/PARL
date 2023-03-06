import parl
import numpy as np

from typing import List
import torch
from benchmark.torch.RL4LMs.utils import  TransitionInfo,\
     RewardFunction
from parl.utils import logger


def compute_batched_rewards(
    episode_wise_transitions: List[List[TransitionInfo]], reward_fn: RewardFunction
):
    # first collect all the prompts, ref and gen texts
    prompts = []
    reference_texts = []
    generated_texts = []
    is_dones = []
    indices = []
    meta_infos = []
    for env_ix, transitions in enumerate(episode_wise_transitions):
        for trans_ix, transition in enumerate(transitions):
            done = transition.done
            info = transition.info
            prompts.append(info["prompt_text"])
            reference_texts.append(info["reference_text"])
            generated_texts.append(info["output"])
            is_dones.append(done)
            meta_infos.append(info["meta_info"])
            indices.append((env_ix, trans_ix))

    # compute rewards all at once
    rewards = reward_fn(prompts, generated_texts, reference_texts, is_dones, meta_infos)
    # rewards = rewards.numpy().flatten()

    # override the rewards in transitions
    for (env_ix, trans_ix), reward in zip(indices, rewards):
        episode_wise_transitions[env_ix][trans_ix].task_reward = reward
        episode_wise_transitions[env_ix][trans_ix].total_reward = (
            reward + episode_wise_transitions[env_ix][trans_ix].kl_reward
        )


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
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


class RL4LMsSummaAgent(parl.Agent):
    def __init__(self,
                 algorithm,
                 alg_config,
                 norm_reward: bool = False,
                 ):
        super(RL4LMsSummaAgent, self).__init__(algorithm)
        self.dataset = None
        self.config = alg_config
        self.n_epochs = alg_config["args"]["n_epochs"]
        self._norm_reward = norm_reward
        self._n_updates = 0




    def learn(self, rollout_buffer):
        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        approx_kl_divs = []
        log_info = {
            "entropy_losses": entropy_losses,
            "pg_losses": pg_losses,
            "value_losses": value_losses,
            "clip_fractions": clip_fractions,
            "approx_kl_divs": approx_kl_divs
        }

        continue_training = True
        loss = torch.tensor(0.0)

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            continue_training, loss = self.alg.learn(rollout_buffer=rollout_buffer,
                                                              log_info=log_info)
            if not continue_training:
                print(
                        f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_divs[-1]:.2f}")
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(
            rollout_buffer.values.flatten(), rollout_buffer.returns.flatten())

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
        train_info["train/clip_range"] =  self.alg.clip_range

        logger.info(train_info)

        ppo_train_info = {
            "ppo/entropy_loss":  np.mean(entropy_losses).item(),
            "ppo/policy_gradient_loss": np.mean(pg_losses).item(),
            "ppo/value_loss": np.mean(value_losses).item(),
            "ppo/approx_kl": np.mean(approx_kl_divs).item(),
        }

        logger.info(ppo_train_info)
        # for k, v in train_info.items():
        #     print(f"{k}: {v}")

    def predict(self, *args, **kwargs):
        pass

    def sample(self, *args, **kwargs):
        pass


