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

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List
from transformers import AutoTokenizer
from copy import deepcopy
from typing import NamedTuple
import torch
import numpy as np

from typing import Any, Union

TensorDict = Dict[Union[str, int], torch.Tensor]


@dataclass
class TransitionInfo(object):
    observation: TensorDict
    action: np.ndarray
    task_reward: np.ndarray
    total_reward: np.ndarray
    kl_div: np.ndarray
    episode_start: np.ndarray
    value: torch.Tensor
    log_prob: torch.Tensor
    done: np.ndarray
    ref_log_prob: torch.Tensor
    kl_reward: np.ndarray
    info: Dict[str, Any]


class DictRolloutBufferSamples(NamedTuple):
    observations: TensorDict
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


@dataclass(init=True)
class Sample(object):
    id: str
    prompt_or_input_text: str
    references: List[str]
    meta_data: Dict[str, Any] = None


class PolicyType(Enum):
    CAUSAL = 0
    SEQ2SEQ = 1


@dataclass
class RefPolicyOutput(object):
    """
    Dataclass for the output of the method policy.get_ref_log_probs()
    """

    # ref log_probs for corresponding observation and chosen action
    log_probs: torch.tensor
    # cached policy activations for sequential forward passes
    past_model_kwargs: torch.tensor


@dataclass
class GenerationInputs(object):
    # prompt inputs
    inputs: torch.tensor
    # prompt attention masks
    attention_masks: torch.tensor


@dataclass
class GenerationOutputs(object):
    # log probs at each time step
    step_wise_logprobs: List[List[torch.tensor]]
    # actions at each time step
    step_wise_actions: List[torch.tensor]
    # generated tokens
    gen_tokens: List[List[int]]
    # generated texts
    gen_texts: List[str]
    # action masks
    action_masks: List[torch.tensor] = None


@dataclass
class Observation(object):
    # encoded input
    prompt_or_input_encoded_pt: torch.tensor
    # attention mask for the input
    prompt_or_input_attention_mask_pt: torch.tensor
    # input text
    prompt_or_input_text: str
    # encoded context
    context_encoded_pt: torch.tensor
    # attention mask for the context
    context_attention_mask_pt: torch.tensor
    # context text
    context_text: str
    # reference texts
    target_or_reference_texts: List[str]

    # concatenated input
    input_encoded_pt: torch.tensor
    input_attention_mask_pt: torch.tensor

    # list of actions
    action_history: List[str]

    # other meta info
    meta_info: Dict[str, Any]

    def to_dict(self):
        """
        For stable baselines (only return tensor items)
        """
        dict_obs = {
            "prompt_or_input_encoded_pt": self.prompt_or_input_encoded_pt.numpy().flatten(),
            "prompt_or_input_attention_mask_pt": self.prompt_or_input_attention_mask_pt.numpy().flatten(),
            "context_encoded_pt": self.context_encoded_pt.numpy().flatten(),
            "context_attention_mask_pt": self.context_attention_mask_pt.numpy().flatten(),
            "input_encoded_pt": self.input_encoded_pt.numpy().flatten(),
            "input_attention_mask_pt": self.input_attention_mask_pt.numpy().flatten()
        }
        return dict_obs

    @staticmethod
    def _concat(prompt: torch.tensor, prompt_mask: torch.tensor, context: torch.tensor, context_mask: torch.tensor,
                pad_token: int):

        prompt_ = prompt[:, prompt_mask.flatten().bool().tolist()]
        context_ = context[:, context_mask.flatten().bool().tolist()]
        actual_size = prompt_.shape[1] + context_.shape[1]

        full_size = prompt.shape[1] + context.shape[1]
        concatenated = torch.full((full_size, ), fill_value=pad_token).reshape(1, -1)
        concatenated_mask = torch.zeros((1, full_size)).int()

        concatenated[:, full_size - actual_size:] = torch.cat((prompt_, context_), dim=1)
        concatenated_mask[:, full_size - actual_size:] = 1
        return concatenated, concatenated_mask

    def update(self, action: int, tokenizer: AutoTokenizer):
        """
        Updates the observation using the given action
        """

        # update the action history
        current_action_history = deepcopy(self.action_history)
        current_action_history.append(tokenizer._convert_id_to_token(action))

        # get the current context
        current_context = deepcopy(self.context_encoded_pt)
        current_context_attention_mask = deepcopy(self.context_attention_mask_pt)

        # just shift the context (also the attention mask) to left by 1
        current_context[:, 0:-1] = current_context[:, 1:].clone()
        current_context_attention_mask[:, 0:-1] = current_context_attention_mask[:, 1:].clone()

        # add the action always at the end (assumes left padding)
        current_context[:, -1] = action
        current_context_attention_mask[:, -1] = 1

        # decode the context
        context_text = tokenizer.decode(current_context.flatten(), skip_special_tokens=True)

        # concatenate and still keep the left padding
        input_encoded_pt, input_attention_mask_pt = Observation._concat(
            self.prompt_or_input_encoded_pt, self.prompt_or_input_attention_mask_pt, current_context,
            current_context_attention_mask, tokenizer.pad_token_id)

        # and create a new observation
        obs = Observation(self.prompt_or_input_encoded_pt, self.prompt_or_input_attention_mask_pt,
                          self.prompt_or_input_text, current_context, current_context_attention_mask, context_text,
                          self.target_or_reference_texts, input_encoded_pt, input_attention_mask_pt,
                          current_action_history, self.meta_info)

        return obs

    @classmethod
    def init_from_sample(cls,
                         sample: Sample,
                         tokenizer: AutoTokenizer,
                         max_input_length: int,
                         max_context_length: int,
                         prompt_truncation_side: str,
                         context_start_token: int = None,
                         meta_info: Dict[str, Any] = None):
        # encode the prompt text
        # override truncation side for prompt
        prev_truncation_side = tokenizer.truncation_side
        tokenizer.truncation_side = prompt_truncation_side
        prompt_outputs = tokenizer(
            sample.prompt_or_input_text,
            padding="max_length",
            max_length=max_input_length,
            return_tensors="pt",
            return_attention_mask=True,
            truncation=True)
        tokenizer.truncation_side = prev_truncation_side

        # for seq2seq models, context should be initialized to start token if provided
        if context_start_token is not None:
            context_outputs = tokenizer(
                "",
                padding="max_length",
                max_length=max_context_length,
                return_tensors="pt",
                return_attention_mask=True)
            context_outputs.input_ids = torch.ones(1, max_context_length, dtype=torch.int32) * tokenizer.pad_token_id
            context_outputs.input_ids[:, -1] = context_start_token
            context_outputs.attention_mask = torch.zeros(1, max_context_length, dtype=torch.int32)
            context_outputs.attention_mask[:, -1] = 1
        else:
            context_outputs = tokenizer(
                "",
                padding="max_length",
                max_length=max_context_length,
                return_tensors="pt",
                return_attention_mask=True)

        # concatenate
        input_encoded_pt, input_attention_mask_pt = Observation._concat(
            prompt_outputs.input_ids, prompt_outputs.attention_mask, context_outputs.input_ids,
            context_outputs.attention_mask, tokenizer.pad_token_id)

        obs = Observation(
            prompt_or_input_encoded_pt=prompt_outputs.input_ids,
            prompt_or_input_attention_mask_pt=prompt_outputs.attention_mask,
            prompt_or_input_text=sample.prompt_or_input_text,
            context_encoded_pt=context_outputs.input_ids,
            context_attention_mask_pt=context_outputs.attention_mask,
            input_encoded_pt=input_encoded_pt,
            input_attention_mask_pt=input_attention_mask_pt,
            context_text="",
            target_or_reference_texts=sample.references,
            action_history=[],
            meta_info=meta_info)

        return obs
