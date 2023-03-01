from abc import abstractmethod, ABC
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
from gym.spaces import Discrete
from gym.spaces.dict import Dict as DictSpace
from torch.distributions import Categorical
from transformers import AutoTokenizer, PreTrainedModel
from transformers.modeling_utils import unwrap_model
from torch import nn

import gym
import numpy as np

from benchmark.torch.RL4LMs.utils import (
    Schedule, TensorDict, 
    
    CategoricalDistribution,

    EvaluateActionsOutput, PolicyOutput, RefPolicyOutput, ValueOutput, 
    GenerationInputs, GenerationOutputs, PolicyType
)


# refer to stable_baselines3.common.policies
class BaseModel(nn.Module, ABC):
    """
    The base model object: makes predictions in response to observations.

    In the case of policies, the prediction is an action. In the case of critics, it is the
    estimated value of the observation.

    :param observation_space: The observation space of the environment
    :param action_space: The action space of the environment
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``torch.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        # features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        features_extractor: Optional[nn.Module] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}

        self.observation_space = observation_space
        self.action_space = action_space
        self.features_extractor = features_extractor
        self.normalize_images = normalize_images

        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer = None  # type: Optional[torch.optim.Optimizer]

        # self.features_extractor_class = features_extractor_class
        self.features_extractor_kwargs = features_extractor_kwargs

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    # def _update_features_extractor(
    #     self,
    #     net_kwargs: Dict[str, Any],
    #     features_extractor: Optional[BaseFeaturesExtractor] = None,
    # ) -> Dict[str, Any]:
    #     """
    #     Update the network keyword arguments and create a new features extractor object if needed.
    #     If a ``features_extractor`` object is passed, then it will be shared.
    #
    #     :param net_kwargs: the base network keyword arguments, without the ones
    #         related to features extractor
    #     :param features_extractor: a features extractor object.
    #         If None, a new object will be created.
    #     :return: The updated keyword arguments
    #     """
    #     net_kwargs = net_kwargs.copy()
    #     if features_extractor is None:
    #         # The features extractor is not shared, create a new one
    #         features_extractor = self.make_features_extractor()
    #     net_kwargs.update(dict(features_extractor=features_extractor, features_dim=features_extractor.features_dim))
    #     return net_kwargs
    #
    # def make_features_extractor(self) -> BaseFeaturesExtractor:
    #     """Helper method to create a features extractor."""
    #     return self.features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
    #
    # def extract_features(self, obs: torch.Tensor) -> torch.Tensor:
    #     """
    #     Preprocess the observation if needed and extract features.
    #
    #     :param obs:
    #     :return:
    #     """
    #     assert self.features_extractor is not None, "No features extractor was set"
    #     preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
    #     return self.features_extractor(preprocessed_obs)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        """
        Get data that need to be saved in order to re-create the model when loading it from disk.

        :return: The dictionary to pass to the as kwargs constructor when reconstruction this model.
        """
        return dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            # Passed to the constructor by child class
            # squash_output=self.squash_output,
            # features_extractor=self.features_extractor
            normalize_images=self.normalize_images,
        )

    # @property
    # def device(self) -> torch.device:
    #     """Infer which device this policy lives on by inspecting its parameters.
    #     If it has no parameters, the 'cpu' device is used as a fallback.
    #
    #     :return:"""
    #     for param in self.parameters():
    #         return param.device
    #     return get_device("cpu")

    def save(self, path: str) -> None:
        """
        Save model to a given location.

        :param path:
        """
        torch.save({"state_dict": self.state_dict(), "data": self._get_constructor_parameters()}, path)

    # @classmethod
    # def load(cls, path: str, device: Union[torch.device, str] = "auto") -> "BaseModel":
    #     """
    #     Load model from patorch.
    #
    #     :param path:
    #     :param device: Device on which the policy should be loaded.
    #     :return:
    #     """
    #     device = get_device(device)
    #     saved_variables = torch.load(path, map_location=device)
    #
    #     # Allow to load policy saved with older version of SB3
    #     if "sde_net_arch" in saved_variables["data"]:
    #         warnings.warn(
    #             "sde_net_arch is deprecated, please downgrade to SB3 v1.2.0 if you need such parameter.",
    #             DeprecationWarning,
    #         )
    #         del saved_variables["data"]["sde_net_arch"]
    #
    #     # Create policy object
    #     model = cls(**saved_variables["data"])  # pytype: disable=not-instantiable
    #     # Load weights
    #     model.load_state_dict(saved_variables["state_dict"])
    #     model.to(device)
    #     return model

    def load_from_vector(self, vector: np.ndarray) -> None:
        """
        Load parameters from a 1D vector.

        :param vector:
        """
        torch.nn.utils.vector_to_parameters(torch.FloatTensor(vector).to(self.device), self.parameters())

    def parameters_to_vector(self) -> np.ndarray:
        """
        Convert the parameters to a 1D vector.

        :return:
        """
        return torch.nn.utils.parameters_to_vector(self.parameters()).detach().cpu().numpy()

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.train(mode)
    #
    # def obs_to_tensor(self, observation: Union[np.ndarray, Dict[str, np.ndarray]]) -> Tuple[torch.Tensor, bool]:
    #     """
    #     Convert an input observation to a PyTorch tensor that can be fed to a model.
    #     Includes sugar-coating to handle different observations (e.g. normalizing images).
    #
    #     :param observation: the input observation
    #     :return: The observation as PyTorch tensor
    #         and whether the observation is vectorized or not
    #     """
    #     vectorized_env = False
    #     if isinstance(observation, dict):
    #         # need to copy the dict as the dict in VecFrameStack will become a torch tensor
    #         observation = copy.deepcopy(observation)
    #         for key, obs in observation.items():
    #             obs_space = self.observation_space.spaces[key]
    #             if is_image_space(obs_space):
    #                 obs_ = maybe_transpose(obs, obs_space)
    #             else:
    #                 obs_ = np.array(obs)
    #             vectorized_env = vectorized_env or is_vectorized_observation(obs_, obs_space)
    #             # Add batch dimension if needed
    #             observation[key] = obs_.reshape((-1,) + self.observation_space[key].shape)
    #
    #     elif is_image_space(self.observation_space):
    #         # Handle the different cases for images
    #         # as PyTorch use channel first format
    #         observation = maybe_transpose(observation, self.observation_space)
    #
    #     else:
    #         observation = np.array(observation)
    #
    #     if not isinstance(observation, dict):
    #         # Dict obs need to be handled separately
    #         vectorized_env = is_vectorized_observation(observation, self.observation_space)
    #         # Add batch dimension if needed
    #         observation = observation.reshape((-1,) + self.observation_space.shape)
    #
    #     observation = obs_as_tensor(observation, self.device)
    #     return observation, vectorized_env


class BasePolicy(BaseModel):
    """The base policy object.

    Parameters are mostly the same as `BaseModel`; additions are documented below.

    :param args: positional arguments passed through to `BaseModel`.
    :param kwargs: keyword arguments passed through to `BaseModel`.
    :param squash_output: For continuous actions, whether the output is squashed
        or not using a ``tanh()`` function.
    """

    def __init__(self, *args, squash_output: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._squash_output = squash_output

    @staticmethod
    def _dummy_schedule(progress_remaining: float) -> float:
        """(float) Useful for pickling policy."""
        del progress_remaining
        return 0.0

    @property
    def squash_output(self) -> bool:
        """(bool) Getter for squash_output."""
        return self._squash_output

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    @abstractmethod
    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Get the action according to the policy for a given observation.

        By default provides a dummy implementation -- not all BasePolicy classes
        implement this, e.g. if they are a Critic in an Actor-Critic method.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # TODO (GH/1): add support for RNN policies
        # if state is None:
        #     state = self.initial_state
        # if episode_start is None:
        #     episode_start = [False for _ in range(self.n_envs)]
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        with torch.no_grad():
            actions = self._predict(observation, deterministic=deterministic)
        # Convert to numpy
        actions = actions.cpu().numpy()

        if isinstance(self.action_space, gym.spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions[0]

        return actions, state

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)

        :param action: Action to scale
        :return: Scaled action
        """
        low, high = self.action_space.low, self.action_space.high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param scaled_action: Action to un-scale
        """
        low, high = self.action_space.low, self.action_space.high
        return low + (0.5 * (scaled_action + 1.0) * (high - low))

class LMActorCriticPolicy(BasePolicy):
    def __init__(
        self,
        observation_space: DictSpace,
        action_space: Discrete,
        lr_schedule: Schedule,
        model_name: str,
        optimizer_kwargs: Dict[str, Any] = {},
        weight_decay: float = 1e-6,
        use_sde: bool = None,
        apply_model_parallel: bool = True,
        optimizer_class: torch.optim.Optimizer = torch.optim.AdamW,
        generation_kwargs: Dict[str, Any] = {},
        prompt_truncation_side: str = "left",
    ):
        """

        Args:
            observation_space (DictSpace): Observation space
            action_space (Discrete): Action space
            lr_schedule (Schedule): Learning rate schedule
            model_name (str): name of the causal or seq2seq model from transformers library
            optimizer_kwargs (Dict[str, Any], optional): optimizer kwargs. Defaults to {}.
            weight_decay (float, optional): weight decay. Defaults to 1e-6.
            use_sde (bool, optional): Use state-dependent exploration. Defaults to None. (Unused parameter from stable-baselines3)
            apply_model_parallel (bool, optional): whether to apply model parallel. Defaults to True.
            optimizer_class (torch.optim.Optimizer, optional): Optimizer class. Defaults to torch.optim.AdamW.
            generation_kwargs (Dict[str, Any], optional): generation parameters for rollout. Defaults to {}.
            prompt_truncation_side (str, optional): truncation side for prompt text. Defaults to "left".
        """
        super().__init__(observation_space, action_space)
        self._action_space = action_space
        self._apply_model_parallel = apply_model_parallel
        self._build_model_heads(model_name)
        self._setup_optimizer(optimizer_kwargs, weight_decay, optimizer_class)
        self._action_dist = CategoricalDistribution(self._action_space.n)
        self._generation_kwargs = generation_kwargs
        self._prompt_truncation_side = prompt_truncation_side

    def _setup_optimizer(
        self,
        optimizer_kwargs: Dict[str, Any],
        weight_decay: float,
        optimizer_class: torch.optim,
    ):
        params = list(self.named_parameters())

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in params if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = optimizer_class(
            optimizer_grouped_parameters, **optimizer_kwargs
        )

    def forward(self, *args, **kwargs):
        # dummy just to comply with base policy
        pass

    @staticmethod
    def _predict(
        self, observation: Dict[str, torch.tensor], deterministic: bool = False
    ) -> torch.Tensor:
        # dummy just to comply with base policy
        pass

    def is_encoder_decoder(self, model: PreTrainedModel):
        return unwrap_model(model).config.is_encoder_decoder

    def generate(
        self,
        tokenizer: AutoTokenizer,
        texts: List[str] = None,
        max_prompt_length: int = None,
        input_ids: torch.tensor = None,
        attention_mask: torch.tensor = None,
        gen_kwargs: Dict[str, Any] = None,
    ) -> GenerationOutputs:

        # if it different from rollout gen kwargs
        if gen_kwargs is None:
            gen_kwargs = self._generation_kwargs

        # switch to eval
        self._policy_model.eval()

        if (
            input_ids is None
            and attention_mask is None
            and texts is not None
            and max_prompt_length is not None
        ):
            # override truncation side for prompt
            prev_truncation_side = tokenizer.truncation_side
            tokenizer.truncation_side = self._prompt_truncation_side
            encodings = tokenizer(
                texts,
                padding="max_length",
                max_length=max_prompt_length,
                return_tensors="pt",
                return_attention_mask=True,
                truncation=True,
            )
            input_ids = encodings.input_ids
            attention_mask = encodings.attention_mask
            tokenizer.truncation_side = prev_truncation_side

        # if min_length argument is set and if policy is not a seq2seq LM (ie. causal LM)
        # then it has to be adjusted to input_size + min_length
        if "min_length" in gen_kwargs.keys() and not self.is_encoder_decoder(
            self._policy_model
        ):
            generation_kwargs_ = deepcopy(gen_kwargs)
            generation_kwargs_["min_length"] = (
                input_ids.shape[1] + gen_kwargs["min_length"]
            )
        else:
            generation_kwargs_ = gen_kwargs

        # generate
        gen_output = unwrap_model(self._policy_model).generate(
            inputs=input_ids.to(self.get_policy_first_device()),
            attention_mask=attention_mask.to(self.get_policy_first_device()),
            return_dict_in_generate=True,
            output_scores=True,
            **generation_kwargs_,
        )

        # number of tokens generated
        seq_length = len(gen_output["scores"])

        # get only the generated text (excluding prompt)
        gen_tokens = gen_output["sequences"][:, -seq_length:]

        # to texts
        gen_texts = [
            tokenizer.decode(output, skip_special_tokens=True)
            for output in gen_tokens.tolist()
        ]

        # extract scores (logits)
        step_wise_logprobs = []
        step_wise_actions = []
        for step, logits in enumerate(gen_output["scores"]):
            raw_logits, _ = logits
            actions_at_step = gen_tokens[:, step]
            distribution = Categorical(logits=raw_logits)
            log_probs = distribution.log_prob(actions_at_step)
            step_wise_logprobs.append(log_probs)
            step_wise_actions.append(actions_at_step)

        gen_output = GenerationOutputs(
            step_wise_logprobs, step_wise_actions, gen_tokens, gen_texts
        )
        return gen_output

    def get_language_model(self):
        return unwrap_model(self._policy_model)

    # Following methods need to be implemented by sub-classing
    @abstractmethod
    def _build_model_heads(self, model_name: str):
        """
        Builds policy and value models
        and sets self._policy_model and self._value_model
        """
        raise NotImplementedError

    @abstractmethod
    def forward_policy(
        self,
        obs: TensorDict,
        actions: torch.tensor,
        past_model_kwargs: Optional[Dict[str, torch.tensor]] = None,
    ) -> PolicyOutput:
        """
        Performs a forward pass on the policy and gets log_probs, entropy etc
        corresponding to specified observation, actions

        This is invoked during rollout generation

        Args:
            obs (TensorDict): observation
            actions (torch.tensor): actions
            past_model_kwargs (Optional[Dict[str, torch.tensor]], optional): Any cached past model activations which can be used for sequential foward passes.
            Defaults to None.
        """
        raise NotImplementedError

    @abstractmethod
    def forward_value(
        self,
        obs: TensorDict,
        past_model_kwargs: Optional[Dict[str, torch.tensor]] = None,
    ) -> ValueOutput:
        """
        Performs a forward pass on the value network and gets values corresponding to observations

        This is invoked during rollout generation

        Args:
            obs (TensorDict): observation
            past_model_kwargs (Optional[Dict[str, torch.tensor]], optional): Any cached past model activations which can be used for sequential foward passes.
            Defaults to None.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> EvaluateActionsOutput:
        """
        Evaluates specified <observation, action>
        and returns log_probs, values, entropy

        This is invoked for each mini-batch in rollout buffer during training iteration
        """
        raise NotImplementedError

    @abstractmethod
    def get_log_probs_ref_model(
        self,
        obs: TensorDict,
        action: torch.tensor,
        past_model_kwargs: Dict[str, Any] = None,
    ) -> RefPolicyOutput:
        """
        Performs a forward pass on the reference policy and gets log_probs
        corresponding to specified observation, actions

        This is invoked during rollout generation to compute KL rewards

        Args:
            obs (TensorDict): observation
            past_model_kwargs (Optional[Dict[str, torch.tensor]], optional): Any cached past model activations which can be used for sequential foward passes.
            Defaults to None.
        """
        raise NotImplementedError

    @abstractmethod
    def get_policy_first_device(self) -> torch.device:
        """
        Returns the first device of the policy. Used in the case of model parallel
        """
        raise NotImplementedError

    @abstractmethod
    def get_policy_type(self) -> PolicyType:
        """
        Returns the type of policy (causal or seq2seq)
        """
        raise NotImplementedError

    @abstractmethod
    def get_inputs_for_generation(self, obs: TensorDict) -> GenerationInputs:
        """
        Extracts the prompt inputs and attention masks which is used as seed for generation
        """
        raise NotImplementedError
