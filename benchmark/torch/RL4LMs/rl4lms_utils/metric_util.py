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

import torch
import numpy as np
from datasets import load_metric
from gem_metrics.msttr import MSTTR
from gem_metrics.ngrams import NGramStats
from gem_metrics.texts import Predictions
from parl.utils import logger


class MeteorMetric(object):
    def __init__(self):
        super().__init__()
        self._metric = load_metric("meteor")

    def compute(
            self,
            prompt_texts,
            generated_texts,
            reference_texts,
            meta_infos=None,
            model=None,
            split_name=None,
    ):

        score = self._metric.compute(predictions=generated_texts, references=reference_texts)["meteor"]

        metric_dict = {"lexical/meteor": (None, score)}
        return metric_dict


class RougeMetric(object):
    def __init__(self, use_single_ref=True):
        super().__init__()
        self._metric = load_metric("rouge")
        self._use_single_ref = use_single_ref

    def compute(
            self,
            prompt_texts,
            generated_texts,
            reference_texts,
            meta_infos=None,
            model=None,
            split_name=None,
    ):
        if self._use_single_ref:
            # TBD: this is required for CNN/DM dataset, without this we get low scores
            # TBD: needs investigation
            ref_texts = [ref[0] for ref in reference_texts]
        else:
            ref_texts = reference_texts

        metric_results = self._metric.compute(predictions=generated_texts, references=ref_texts, use_stemmer=True)
        score_keys = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        metric_dict = {}
        for rouge_type in score_keys:
            rouge_score = metric_results[rouge_type].mid.fmeasure
            metric_dict[f"lexical/rouge_{rouge_type}"] = (None, rouge_score)
        return metric_dict


class BERTScoreMetric(object):
    def __init__(self, language):
        super().__init__()
        self._metric = load_metric("bertscore")
        self._language = language
        # since models are loaded heavily on cuda:0, use the last one to avoid memory
        self._last_gpu = f"cuda:{torch.cuda.device_count() - 1}"

    def compute(
            self,
            prompt_texts,
            generated_texts,
            reference_texts,
            meta_infos=None,
            model=None,
            split_name=None,
    ):
        with torch.no_grad():
            metric_results = self._metric.compute(
                predictions=generated_texts,
                references=reference_texts,
                lang=self._language,
                device=self._last_gpu,
            )
            bert_scores = metric_results["f1"]
            corpus_level_score = np.mean(bert_scores)
            metric_dict = {"semantic/bert_score": (bert_scores, corpus_level_score)}
            return metric_dict


class BLEUMetric(object):
    def __init__(self):
        super().__init__()
        self._metric = load_metric("bleu")

    def compute(
            self,
            prompt_texts,
            generated_texts,
            reference_texts,
            meta_infos=None,
            model=None,
            split_name=None,
    ):

        tokenized_predictions = []
        tokenized_reference_texts = []
        for prediction, refs in zip(generated_texts, reference_texts):
            tokenized_prediction = prediction.split()
            tokenized_refs = [ref.split() for ref in refs]
            tokenized_predictions.append(tokenized_prediction)
            tokenized_reference_texts.append(tokenized_refs)

        try:
            metric_results = self._metric.compute(
                predictions=tokenized_predictions, references=tokenized_reference_texts)
            bleu_score = metric_results["bleu"]
            metric_dict = {"lexical/bleu": (None, bleu_score)}
            return metric_dict
        except Exception as e:
            return {"lexical/bleu": (None, "n/a")}


class DiversityMetrics(object):
    def __init__(self, window_size=100):
        self._msttr_metric = MSTTR(window_size=window_size)
        self._n_gram_metric = NGramStats()

    def compute(
            self,
            prompt_texts,
            generated_texts,
            reference_texts,
            meta_infos=None,
            model=None,
            split_name=None,
    ):

        predictions = Predictions(data={"filename": "", "values": generated_texts})
        diversity_metrics = {}
        msttr_metrics = self._msttr_metric.compute(None, predictions)
        n_gram_metrics = self._n_gram_metric.compute(None, predictions)

        for key, value in msttr_metrics.items():
            diversity_metrics[f"diversity_metrics/{key}"] = (None, value)
        for key, value in n_gram_metrics.items():
            diversity_metrics[f"diversity_metrics/{key}"] = (None, value)

        return diversity_metrics


class MetricRegistry(object):
    _registry = {
        "meteor": MeteorMetric,
        "rouge": RougeMetric,
        "bert_score": BERTScoreMetric,
        "bleu": BLEUMetric,
        "diversity": DiversityMetrics,
    }

    @classmethod
    def get(cls, metric_id, kwargs):
        logger.info(f"loading metric: {metric_id}")
        metric_cls = cls._registry[metric_id]
        metric = metric_cls(**kwargs)
        return metric

    @classmethod
    def add(cls, id, metric_cls):
        MetricRegistry._registry[id] = metric_cls
