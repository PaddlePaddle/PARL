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

from datasets import load_metric


class RougeRewardFunction(object):
    def __init__(self, rouge_type, use_single_ref=True):
        super().__init__()
        self._metric = load_metric("rouge")
        self._rouge_type = rouge_type

        self._shaping_fn = None
        self._use_single_ref = use_single_ref

    def __call__(
            self,
            current_observation,
            action,
            next_observation,
            done,
            meta_info=None,
    ):
        if done:
            # TBD: considers only one reference for now
            if self._use_single_ref:
                references = [next_observation.target_or_reference_texts[0]]
            else:
                references = [next_observation.target_or_reference_texts]
            predicted = [next_observation.context_text]

            metric_results = self._metric.compute(predictions=predicted, references=references, use_stemmer=True)
            reward = metric_results[self._rouge_type].mid.fmeasure
            if self._shaping_fn is not None:
                aux_score = self._shaping_fn(current_observation, action, next_observation, done, meta_info)
                reward = reward + aux_score
            return reward
        return 0
