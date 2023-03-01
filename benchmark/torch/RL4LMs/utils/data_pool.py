from datasets import load_dataset
from .data_wrapper import Sample
from typing import Any, List, Dict
import random
from abc import abstractclassmethod
from tqdm import tqdm
from nltk.tokenize import word_tokenize

class TextGenPool:
    def __init__(self, samples: List[Sample]):
        self._samples = samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, ix: int) -> Sample:
        if ix >= len(self):
            raise StopIteration
        sample = self._samples[ix]
        return sample, 1.0

    def sample(self) -> Sample:
        random_sample = random.choice(self._samples)
        return random_sample

    @abstractclassmethod
    def prepare(cls, **args) -> 'TextGenPool':
        """
        A factory method to instantiate data pool
        """
        raise NotImplementedError

    def split(self, split_ratios: List[float]) -> List['TextGenPool']:
        start_ix = 0
        pools = []
        for ratio in split_ratios:
            count = int(len(self) * ratio)
            end_ix = start_ix + count
            pools.append(type(self)(self._samples[start_ix: end_ix]))
            start_ix = end_ix
        return pools

class CommonGen(TextGenPool):
    @classmethod
    def prepare(cls, split: str,
                concept_separator_token: str = " ",
                concept_end_token=" ",
                prefix: str = "summarize: ") -> 'TextGenPool':
        ds = load_dataset("gem", "common_gen")
        samples = []
        split_id = CommonGen.gen_split_name(split)
        for ix, item in enumerate(ds[split_id]):
            concepts = concept_separator_token.join(item["concepts"])
            concepts = prefix + concepts
            concepts += concept_end_token
            if item["target"] == "":
                # just to avoid breaking of metric computation
                item["target"] = "empty reference"
            targets = [item["target"]]
            sample = Sample(id=f"{split}_{ix}",
                            prompt_or_input_text=concepts,
                            references=targets,
                            meta_data={
                                "concepts": item["concepts"]
                            }
                            )
            samples.append(sample)
        pool_instance = cls(samples)
        return pool_instance

    @staticmethod
    def gen_split_name(split: str):
        if split == "train":
            split_name = "train"
        elif split == "val":
            split_name = "validation"
        elif split == "test":
            split_name = "test"
        else:
            raise NotImplementedError
        return split_name



class CNNDailyMail(TextGenPool):
    @classmethod
    def prepare(cls,
                split: str,
                prompt_suffix: str = "",
                prompt_prefix: str = "",
                truncate_article: int = None,
                max_size: int = None):
        dataset = load_dataset("cnn_dailymail", "3.0.0")
        dataset_split = CommonGen.gen_split_name(split)
        samples = []
        for ix, item in tqdm(enumerate(dataset[dataset_split]),
                             desc="Tokenizing dataset",
                             total=len(dataset[dataset_split])):

            if truncate_article is not None:
                tokens = word_tokenize(item["article"])
                tokens = tokens[:truncate_article]
                item["article"] = " ".join(tokens)

            sample = Sample(id=f"{split}_{ix}",
                            prompt_or_input_text=prompt_prefix +
                            item["article"] + prompt_suffix,
                            references=[item["highlights"]]
                            )
            samples.append(sample)

            if max_size is not None and ix == (max_size-1):
                break

        pool_instance = cls(samples)
        return pool_instance