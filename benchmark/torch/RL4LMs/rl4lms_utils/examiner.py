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

from tqdm import tqdm
from parl.utils import logger


# class for results evaluation
class Examiner(object):
    def __init__(self, tokenizer, eval_batch_size, metrics, eval_gen_kwargs, samples_by_split, max_prompt_length):
        self._tokenizer = tokenizer
        self._batch_size = eval_batch_size
        self._metrics = metrics
        self._gen_kwargs = eval_gen_kwargs
        self._samples_by_split = samples_by_split
        self._max_prompt_length = max_prompt_length

    def evaluate(self, policy, sample_name_list, epoch):
        for split_name in sample_name_list:
            self._evaluate_on_samples(policy=policy, epoch=epoch, split_name=split_name)

    def _evaluate_on_samples(
            self,
            policy,
            epoch,
            split_name,
            dt_control_token="",
    ):
        samples = self._samples_by_split[split_name]
        # generate text by batch
        all_generated_texts = []
        all_ref_texts = []
        all_prompt_texts = []
        all_meta_infos = []

        n_samples = len(samples)
        for batch in tqdm(list(self._get_batch(samples, self._batch_size)), desc="Evaluating"):
            batch_generated_texts = self._generate_text(policy, self._tokenizer, batch, self._max_prompt_length,
                                                        dt_control_token)
            batch_ref_texts = [sample.references for sample in batch]
            batch_prompt_texts = [sample.prompt_or_input_text for sample in batch]
            batch_meta_infos = [sample.meta_data for sample in batch]
            all_generated_texts.extend(batch_generated_texts)
            all_ref_texts.extend(batch_ref_texts)
            all_prompt_texts.extend(batch_prompt_texts)
            all_meta_infos.extend(batch_meta_infos)

        # compute metrics
        corpus_level_metrics = {}
        sample_scores_by_metric = {}
        if self._metrics is not None:
            for metric in self._metrics:
                metric_dict = metric.compute(
                    all_prompt_texts,
                    all_generated_texts,
                    all_ref_texts,
                    all_meta_infos,
                    policy.get_language_model(),
                    split_name,
                )

                for metric_key, (sample_scores, corpus_score) in metric_dict.items():
                    if sample_scores is None:
                        sample_scores = ["n/a"] * n_samples
                    corpus_level_metrics[metric_key] = corpus_score
                    sample_scores_by_metric[metric_key] = sample_scores

        # aggregate sample metric scores
        sample_predictions_dict = []
        for ix, (sample, prompt_text, generated_text, ref_texts) in enumerate(
                zip(samples, all_prompt_texts, all_generated_texts, all_ref_texts)):
            sample_prediction = {
                "split_name":
                split_name,
                "sample_id":
                sample.id,
                "prompt_text":
                prompt_text,
                "generated_text":
                generated_text,
                "ref_text":
                "".join([
                    f"<START-{ref_ix+1}>" + ref_text + f"<END-{ref_ix+1}>" for ref_ix, ref_text in enumerate(ref_texts)
                ]),
            }
            for metric_key, sample_scores in sample_scores_by_metric.items():
                sample_prediction[metric_key] = sample_scores[ix]
            sample_predictions_dict.append(sample_prediction)

        metrics_dict_ = {"epoch": epoch, "metrics": corpus_level_metrics}

        # logger
        logger.info(f"{split_name} metrics: {metrics_dict_}")

    def _get_batch(self, samples, batch_size):
        current_ix = 0
        n_samples = len(samples)
        while current_ix < n_samples:
            current_batch = samples[current_ix:current_ix + batch_size]
            yield current_batch
            current_ix += batch_size

    def _generate_text(
            self,
            policy,
            tokenizer,
            samples,
            max_prompt_length,
            dt_control_token,
    ):
        prompt_texts = [dt_control_token + sample.prompt_or_input_text for sample in samples]
        generated_texts = policy.predict(
            tokenizer, prompt_texts, max_prompt_length, gen_kwargs=self._gen_kwargs).gen_texts
        return generated_texts
