from typing import Optional, Dict, Any
import torch


class KLController:
    def __init__(self, kl_coeff, target_kl = None):
        self._kl_coeff = kl_coeff
        self._target_kl = target_kl

    def step(self, kl_div: torch.tensor):
        """
        Adapts the KL coeff
        """
        if self._target_kl is not None:
            diff_to_target = (kl_div - self._target_kl) / self._target_kl
            e_t = torch.clip(diff_to_target, -0.2, 0.2).item()
            self._kl_coeff = self._kl_coeff * (1 + 0.1 * e_t)

    @property
    def kl_coeff(self):
        return self._kl_coeff
