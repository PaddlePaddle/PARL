from abc import ABC, abstractclassmethod

import torch
from datasets import load_metric
from .data_wrapper import Observation
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from benchmark.torch.RL4LMs.metrics import (
    MeteorMetric,
    BERTScoreMetric,
    BLEUMetric,
    RougeLMax,
)
import numpy as np
from typing import List, Dict, Any


class RewardFunction(ABC):
    @abstractclassmethod
    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        """
        Callable for reward functions for text generation

        Args:
            current_observation (Observation): previous observation (s)
            action (int): action performed (a) at s
            next_observation (Observation): observation after the action was performed (s')
            done (bool): whether the episode is finished or not
            meta_info (dict) - other information regarding textual sample
        Returns:
            float: scalar reward
        """
        raise NotImplementedError


class BatchedRewardFunction(ABC):
    """
    Computes rewards for several instances at once
    """

    @abstractclassmethod
    def __call__(
        self,
        prompt_texts: List[str],
        gen_texts: List[str],
        ref_texts: List[List[str]],
        dones: List[bool],
        meta_infos: List[Dict[str, Any]] = None,
    ) -> List[float]:
        """
        An abstract class for batched reward functions for text generation
        """
        raise NotImplementedError


### Automated reward functions ###########################


class CommonGenPenaltyShapingFunction(RewardFunction):
    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            prompt_text = next_observation.prompt_or_input_text
            prefix = "generate a sentence with: "
            concept_n_grams = prompt_text.split(prefix)[1][:-1]

            if (
                concept_n_grams.lower() in next_observation.context_text.lower()
                or prefix in next_observation.context_text.lower()
                or "generate" in next_observation.context_text.lower()
                or "sentence" in next_observation.context_text.lower()
            ):
                penalty_score = -1
            else:
                penalty_score = 0
            return penalty_score
        return 0





class MeteorRewardFunction(RewardFunction):
    def __init__(self, shaping_fn: str = None) -> None:
        super().__init__()
        self._metric = MeteorMetric()
        from benchmark.torch.RL4LMs.registry import RewardFunctionRegistry

        self._shaping_fn = (
            RewardFunctionRegistry.get(shaping_fn, {})
            if shaping_fn is not None
            else shaping_fn
        )

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:

        # compute meteor at the end of episode
        if done:
            references = [next_observation.target_or_reference_texts]
            predicted = [next_observation.context_text]
            metric_dict = self._metric.compute(None, predicted, references)
            score = metric_dict["lexical/meteor"][1]

            if self._shaping_fn is not None:
                aux_score = self._shaping_fn(
                    current_observation, action, next_observation, done, meta_info
                )
                score = score + aux_score
            return score
        return 0


class RougeRewardFunction(RewardFunction):
    def __init__(
        self, rouge_type: str, shaping_fn: str = None, use_single_ref: bool = True
    ) -> None:
        super().__init__()
        self._metric = load_metric("rouge")
        self._rouge_type = rouge_type
        from benchmark.torch.RL4LMs.registry import RewardFunctionRegistry

        self._shaping_fn = (
            RewardFunctionRegistry.get(shaping_fn, {})
            if shaping_fn is not None
            else shaping_fn
        )
        self._use_single_ref = use_single_ref

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            # TBD: considers only one reference for now
            if self._use_single_ref:
                references = [next_observation.target_or_reference_texts[0]]
            else:
                references = [next_observation.target_or_reference_texts]
            predicted = [next_observation.context_text]

            metric_results = self._metric.compute(
                predictions=predicted, references=references, use_stemmer=True
            )
            reward = metric_results[self._rouge_type].mid.fmeasure
            if self._shaping_fn is not None:
                aux_score = self._shaping_fn(
                    current_observation, action, next_observation, done, meta_info
                )
                reward = reward + aux_score
            return reward
        return 0


class RougeCombined(RewardFunction):
    def __init__(self, shaping_fn: str = None) -> None:
        super().__init__()
        self._metric = load_metric("rouge")
        from benchmark.torch.RL4LMs.registry import RewardFunctionRegistry

        self._shaping_fn = (
            RewardFunctionRegistry.get(shaping_fn, {})
            if shaping_fn is not None
            else shaping_fn
        )

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            # TBD: considers only one reference for now
            references = [next_observation.target_or_reference_texts[0]]
            predicted = [next_observation.context_text]

            metric_results = self._metric.compute(
                predictions=predicted, references=references, use_stemmer=True
            )

            rouge_keys = ["rouge1", "rouge2", "rougeL"]
            scores = [
                metric_results[rouge_type].mid.fmeasure for rouge_type in rouge_keys
            ]
            reward = np.mean(scores)
            if self._shaping_fn is not None:
                aux_score = self._shaping_fn(
                    current_observation, action, next_observation, done, meta_info
                )
                reward = reward + aux_score
            return reward
        return 0


class BERTScoreRewardFunction(RewardFunction):
    def __init__(self, language: str = "en") -> None:
        super().__init__()
        self._metric = BERTScoreMetric(language)

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            references = [next_observation.target_or_reference_texts]
            predicted = [next_observation.context_text]
            metric_results = self._metric.compute(None, predicted, references)
            bert_score = metric_results["semantic/bert_score"][1]
            return bert_score
        return 0


class BLEURewardFunction(RewardFunction):
    def __init__(self) -> None:
        super().__init__()
        self._metric = BLEUMetric()

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            references = [next_observation.target_or_reference_texts]
            predicted = [next_observation.context_text]
            metric_results = self._metric.compute(None, predicted, references)
            bleu_score = metric_results["lexical/bleu"][1]
            return bleu_score
        return 0


class SacreBleu(RewardFunction):
    def __init__(self, **args) -> None:
        super().__init__()
        self._metric = load_metric("sacrebleu")
        self._args = args

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            references = [next_observation.target_or_reference_texts]
            predicted = [next_observation.context_text]
            metric_results = self._metric.compute(
                predictions=predicted, references=references, **self._args
            )
            return metric_results["score"] / 100
        return 0




#############################################################################

########## Learned Reward Functions##########################################


class LearnedRewardFunction(RewardFunction):
    def __init__(
        self, model_name: str, label_ix: int, include_prompt_for_eval: bool = True
    ) -> None:
        super().__init__()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._metric_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._metric_tokenizer.truncation_side = "left"
        self._metric_model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(self._device)
        self._label_ix = label_ix
        self._include_prompt_for_eval = include_prompt_for_eval

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            generated_text = (
                current_observation.prompt_or_input_text
                if self._include_prompt_for_eval
                else ""
            )
            generated_text += next_observation.context_text

            with torch.no_grad():
                encoded = self._metric_tokenizer(
                    generated_text, return_tensors="pt", truncation=True, padding=True
                )
                outputs = self._metric_model(
                    input_ids=encoded.input_ids.to(self._device),
                    attention_mask=encoded.attention_mask.to(self._device),
                )
                scores = torch.softmax(outputs.logits.flatten(), dim=0)
                score = scores[self._label_ix].item()
                return score
        return 0


class BLEURTRewardFunction(RewardFunction):
    def __init__(self, checkpoint: str = None):
        super().__init__()
        self._metric = load_metric("bleurt", checkpoint=checkpoint)

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            references = [next_observation.target_or_reference_texts]
            predicted = [next_observation.context_text]
            metric_results = self._metric.compute(
                predictions=predicted, references=references
            )
            score = metric_results["scores"][0]
            return score
        return 0


# class PARENTRewardFunction(RewardFunction):
#     """
#     PARENT F1 score as the reward
#     """
#
#     def __init__(self) -> None:
#         super().__init__()
#         self._metric = ParentToTTo()
#
#     def __call__(
#         self,
#         current_observation: Observation,
#         action: int,
#         next_observation: Observation,
#         done: bool,
#         meta_info: Dict[str, Any] = None,
#     ) -> float:
#         if done:
#             generated_texts = [next_observation.context_text]
#             meta_infos = [meta_info]
#             scores = self._metric.compute(None, generated_texts, None, meta_infos)
#             reward = scores["table_to_text/parent_overall_f_score"][0][0]
#             return reward
#         return 0


class RougeLMaxRewardFunction(RewardFunction):
    def __init__(self, **args) -> None:
        super().__init__()
        self._metric = RougeLMax(**args)

    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            references = [next_observation.target_or_reference_texts]
            predicted = [next_observation.context_text]
            meta_infos = [meta_info]
            scores = self._metric.compute(None, predicted, references, meta_infos)
            reward = scores["lexical/rouge_l_max"][0][0]
            return reward
        return 0




if __name__ == "__main__":
    predictions = "hello there general kenobi"
    references = ["hello there general kenobi", "hello there!!"]
    observation = Observation(
        None, None, None, None, None, predictions, references, None, None, None, None
    )

    reward_fn = MeteorRewardFunction()
    print(reward_fn(None, None, observation, True))

    # reward_fn = chrF()
    # print(reward_fn(None, None, observation, True))

    reward_fn = RougeCombined()
    print(reward_fn(None, None, observation, True))

    reward_fn = RougeRewardFunction(rouge_type="rouge1")
    print(reward_fn(None, None, observation, True))

    reward_fn = RougeRewardFunction(rouge_type="rouge2")
    print(reward_fn(None, None, observation, True))

    reward_fn = RougeRewardFunction(rouge_type="rougeL")
    print(reward_fn(None, None, observation, True))

    reward_fn = BERTScoreRewardFunction(language="en")
    print(reward_fn(None, None, observation, True))

    reward_fn = BLEURewardFunction()
    print(reward_fn(None, None, observation, True))

    reward_fn = BLEURTRewardFunction()
    print(reward_fn(None, None, observation, True))
