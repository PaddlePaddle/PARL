from typing import Any, Dict, Optional, List, Union, Callable
import torch


# refer to stable_baselines3.common.type_aliases
TensorDict = Dict[Union[str, int], torch.Tensor]
Schedule = Callable[[float], float]
