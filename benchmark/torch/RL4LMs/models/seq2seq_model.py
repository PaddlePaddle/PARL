from typing import Any, Dict, Optional, List, Union
import torch
from gym.spaces import Discrete
from gym.spaces.dict import Dict as DictSpace
from torch import nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from copy import deepcopy

from transformers.modeling_utils import unwrap_model
from benchmark.torch.RL4LMs.utils import (
    override_generation_routines,

    ActorCriticWarmStartMixin,

    TensorDict, Schedule,

    GenerationInputs, PolicyOutput, RefPolicyOutput, ValueOutput,
    PolicyType, EvaluateActionsOutput, GenerationOutputs,
)

from base_model import LMActorCriticPolicy


class Seq2SeqLMModel(LMActorCriticPolicy, ActorCriticWarmStartMixin):
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
        state_dict: Dict[str, Any] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            model_name,
            optimizer_kwargs,
            weight_decay,
            use_sde,
            apply_model_parallel,
            optimizer_class,
            generation_kwargs,
            prompt_truncation_side,
        )
        self.load_from_dict(state_dict)

    def _build_model_heads(self, model_name: str):
        self._policy_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self._policy_model.__class__ = override_generation_routines(type(self._policy_model))

        self._value_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self._ref_model = deepcopy(self._policy_model).eval()

        self._value_head = nn.Linear(
            self._value_model.config.hidden_size, 1, bias=False
        )

        # apply model parallel
        if torch.cuda.is_available():
            if self._apply_model_parallel and self._policy_model.is_parallelizable:
                self._policy_model.parallelize()
                self._ref_model.parallelize()
                self._value_model.parallelize()
                self._value_head = self._value_head.to(self.device)
            else:  # else defaults to data parallel
                self._policy_model = torch.nn.DataParallel(self._policy_model)
                self._ref_model = torch.nn.DataParallel(self._ref_model)
                self._value_model = torch.nn.DataParallel(self._value_model)
                self._value_head = torch.nn.DataParallel(
                    self._value_head.to(self.device)
                )

    def forward_policy(
        self,
        obs: TensorDict,
        actions: torch.tensor,
        past_model_kwargs: Optional[Dict[str, torch.tensor]] = None,
    ) -> PolicyOutput:

        # Temp workaround for Seq2seq policy
        past_model_kwargs = None

        if past_model_kwargs is None:
            # 1. prepare model inputs
            past_model_kwargs = {
                "attention_mask": obs["prompt_or_input_attention_mask_pt"],
            }
            inputs_tensor, model_input_name, past_model_kwargs = unwrap_model(
                self._policy_model
            )._prepare_model_inputs(
                obs["prompt_or_input_encoded_pt"].int(), None, past_model_kwargs
            )

            # 2. prepare encoder outputs
            past_model_kwargs = unwrap_model(
                self._policy_model
            )._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, past_model_kwargs, model_input_name
            )

            # 3. Prepare input_ids for auto-regressive generation
            input_ids = obs["context_encoded_pt"].int()
            decoder_attn_mask = obs["context_attention_mask_pt"]
        else:
            input_ids = obs["context_encoded_pt"].int()
            decoder_attn_mask = past_model_kwargs.pop("decoder_attention_mask")

        # all set to get into auto-regressive mode
        # prepare all of the model inputs for the decoder
        batch_size = input_ids.shape[0]
        model_inputs = unwrap_model(self._policy_model).prepare_inputs_for_generation(
            input_ids, **past_model_kwargs
        )

        # and forward pass to get next token logits
        outputs = self._policy_model(
            **model_inputs, decoder_attention_mask=decoder_attn_mask, return_dict=True
        )
        next_token_logits = outputs.logits[:, -1, :]

        # get log probs
        dist = self._action_dist.proba_distribution(action_logits=next_token_logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        # update the model kwargs for further generation
        past_model_kwargs = unwrap_model(
            self._policy_model
        )._update_model_kwargs_for_generation(
            outputs,
            past_model_kwargs,
            is_encoder_decoder=unwrap_model(
                self._policy_model
            ).config.is_encoder_decoder,
        )
        past_model_kwargs["decoder_attention_mask"] = torch.cat(
            (decoder_attn_mask, torch.ones(batch_size, 1).to(decoder_attn_mask.device)),
            dim=-1,
        )

        policy_output = PolicyOutput(
            actions, log_prob, log_prob, entropy, past_model_kwargs
        )

        return policy_output

    def forward_value(
        self,
        obs: TensorDict,
        past_model_kwargs: Optional[Dict[str, torch.tensor]] = None,
    ) -> ValueOutput:
        # Temp workaround for Seq2seq policy
        past_model_kwargs = None

        if past_model_kwargs is None:
            # 1. prepare model inputs
            past_model_kwargs = {
                "attention_mask": obs["prompt_or_input_attention_mask_pt"],
            }
            inputs_tensor, model_input_name, past_model_kwargs = unwrap_model(
                self._value_model
            )._prepare_model_inputs(
                obs["prompt_or_input_encoded_pt"].int(), None, past_model_kwargs
            )

            # 2. prepare encoder outputs
            past_model_kwargs = unwrap_model(
                self._value_model
            )._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, past_model_kwargs, model_input_name
            )

            # 3. Prepare input_ids for auto-regressive generation
            input_ids = obs["context_encoded_pt"].int()
            decoder_attn_mask = obs["context_attention_mask_pt"]
        else:
            input_ids = obs["context_encoded_pt"].int()
            decoder_attn_mask = past_model_kwargs.pop("decoder_attention_mask")

        # all set to get into auto-regressive mode
        # prepare all of the model inputs for the decoder
        batch_size = input_ids.shape[0]
        model_inputs = unwrap_model(self._value_model).prepare_inputs_for_generation(
            input_ids, **past_model_kwargs
        )

        # and forrward pass to get hidden states
        outputs = self._value_model(
            **model_inputs,
            output_hidden_states=True,
            decoder_attention_mask=decoder_attn_mask,
            return_dict=True
        )

        # get decoder's last hidden state
        last_tokens_hidden = outputs.decoder_hidden_states[-1][:, -1, :].to(self.device)
        values = self._value_head.forward(last_tokens_hidden)

        # update the model kwargs for further generation
        past_model_kwargs = unwrap_model(
            self._value_model
        )._update_model_kwargs_for_generation(
            outputs,
            past_model_kwargs,
            is_encoder_decoder=unwrap_model(
                self._value_model
            ).config.is_encoder_decoder,
        )
        past_model_kwargs["decoder_attention_mask"] = torch.cat(
            (decoder_attn_mask, torch.ones(batch_size, 1).to(decoder_attn_mask.device)),
            dim=-1,
        )

        value_output = ValueOutput(values, past_model_kwargs)
        return value_output

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> EvaluateActionsOutput:

        policy_outputs = self.forward_policy(obs=obs, actions=actions)
        value_outputs = self.forward_value(obs)

        eval_outputs = EvaluateActionsOutput(
            values=value_outputs.values,
            log_prob=policy_outputs.log_probs,
            entropy=policy_outputs.entropy,
        )
        return eval_outputs

    def to(self, device: str):
        if self._apply_model_parallel:
            self._value_head = self._value_head.to(device)
            return self
        else:
            return super().to(device)

    def get_log_probs_ref_model(
        self,
        obs: TensorDict,
        action: torch.tensor,
        model_kwarpast_model_kwargsgs: Dict[str, Any] = None,
    ) -> RefPolicyOutput:
        # Temp workaround for Seq2seq policy
        past_model_kwargs = None

        if past_model_kwargs is None:
            # 1. prepare model inputs
            past_model_kwargs = {
                "attention_mask": obs["prompt_or_input_attention_mask_pt"],
            }
            inputs_tensor, model_input_name, past_model_kwargs = unwrap_model(
                self._ref_model
            )._prepare_model_inputs(
                obs["prompt_or_input_encoded_pt"].int(), None, past_model_kwargs
            )

            # 2. prepare encoder outputs
            past_model_kwargs = unwrap_model(
                self._ref_model
            )._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, past_model_kwargs, model_input_name
            )

            # 3. Prepare input_ids for auto-regressive generation
            input_ids = obs["context_encoded_pt"].int()
            decoder_attn_mask = obs["context_attention_mask_pt"]
        else:
            input_ids = obs["context_encoded_pt"].int()
            decoder_attn_mask = past_model_kwargs.pop("decoder_attention_mask")

        # all set to get into auto-regressive mode
        # prepare all of the model inputs for the decoder
        batch_size = input_ids.shape[0]
        model_inputs = unwrap_model(self._ref_model).prepare_inputs_for_generation(
            input_ids, **past_model_kwargs
        )

        # and forward pass to get next token logits
        outputs = self._ref_model(
            **model_inputs, decoder_attention_mask=decoder_attn_mask, return_dict=True
        )
        next_token_logits = outputs.logits[:, -1, :]

        # get log probs
        dist = self._action_dist.proba_distribution(action_logits=next_token_logits)
        log_prob = dist.log_prob(action)

        # update the model kwargs for further generation
        past_model_kwargs = unwrap_model(
            self._ref_model
        )._update_model_kwargs_for_generation(
            outputs,
            past_model_kwargs,
            is_encoder_decoder=unwrap_model(self._ref_model).config.is_encoder_decoder,
        )
        past_model_kwargs["decoder_attention_mask"] = torch.cat(
            (decoder_attn_mask, torch.ones(batch_size, 1).to(decoder_attn_mask.device)),
            dim=-1,
        )

        ref_policy_output = RefPolicyOutput(log_prob, past_model_kwargs)

        return ref_policy_output

    def get_policy_first_device(self):
        return (
            self._policy_model.get_encoder().first_device
            if self._apply_model_parallel
            else self.device
        )

    def get_inputs_for_generation(self, obs: TensorDict) -> GenerationInputs:

        generation_inputs = GenerationInputs(
            obs["prompt_or_input_encoded_pt"], obs["prompt_or_input_attention_mask_pt"]
        )
        return generation_inputs

    def get_policy_type(self):
        return PolicyType.SEQ2SEQ
