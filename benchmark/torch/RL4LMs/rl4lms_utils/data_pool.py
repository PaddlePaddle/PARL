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

from datasets import load_dataset
from .data_wrapper import Sample
import random
from tqdm import tqdm
from nltk.tokenize import word_tokenize


class CNNDailyMail(object):
    def __init__(self, samples):
        self._samples = samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, ix):
        if ix >= len(self):
            raise StopIteration
        sample = self._samples[ix]
        return sample, 1.0

    @classmethod
    def prepare(cls, split, prompt_suffix="", prompt_prefix="", truncate_article=None, max_size=None):
        split2name = {"train": "train", "val": "validation", "test": "test"}
        dataset = load_dataset("cnn_dailymail", "3.0.0")
        dataset_split = split2name[split]
        samples = []
        for ix, item in tqdm(
                enumerate(dataset[dataset_split]), desc="Tokenizing dataset", total=len(dataset[dataset_split])):

            if truncate_article is not None:
                tokens = word_tokenize(item["article"])
                tokens = tokens[:truncate_article]
                item["article"] = " ".join(tokens)

            sample = Sample(
                id=f"{split}_{ix}",
                prompt_or_input_text=prompt_prefix + item["article"] + prompt_suffix,
                references=[item["highlights"]])
            samples.append(sample)

            if max_size is not None and ix == (max_size - 1):
                break

        pool_instance = cls(samples)
        return pool_instance

    def sample(self):
        random_sample = random.choice(self._samples)
        return random_sample

    def split(self, split_ratios):
        start_ix = 0
        pools = []
        for ratio in split_ratios:
            count = int(len(self) * ratio)
            end_ix = start_ix + count
            pools.append(type(self)(self._samples[start_ix:end_ix]))
            start_ix = end_ix
        return pools
