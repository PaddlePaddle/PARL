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

from transformers import AutoTokenizer
from parl.utils import logger
from .reward_util import RougeRewardFunction
from .metric_util import MetricRegistry
from .data_pool import CNNDailyMail


def build_tokenizer(tokenizer_config):
    logger.info(f"loading tokenizer of [{tokenizer_config['model_name']}] model")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_config["model_name"])
    except Exception:
        logger.info(f"trying to use local_files to load tokenizer of [{tokenizer_config['model_name']}] model")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_config["model_name"], local_files_only=True)
    if tokenizer.pad_token is None and tokenizer_config.get("pad_token_as_eos_token", True):
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = tokenizer_config.get("padding_side", "left")
    tokenizer.truncation_side = tokenizer_config.get("truncation_side", "left")
    return tokenizer


def build_reward_fn(reward_config):
    logger.info(f"loading reward function: rouge")
    reward_fn = RougeRewardFunction(rouge_type=reward_config["rouge_type"])
    return reward_fn


def build_metrics(metric_configs):
    metrics = [
        MetricRegistry.get(metric_config["id"], metric_config.get("args", {})) for metric_config in metric_configs
    ]
    return metrics


def build_datapool(datapool_config, remote_train=False):
    def _get_datapool_by_split(split):
        kwargs = {"prompt_prefix": datapool_config["prompt_prefix"], "split": split}
        logger.info(f"loading split of dataset: {datapool_config['id']} -- {kwargs['split']}")
        dp_split = CNNDailyMail.prepare(split=kwargs["split"], prompt_prefix=kwargs["prompt_prefix"])
        logger.info(f"finish loading split of dataset: {datapool_config['id']} -- {kwargs['split']}")
        return dp_split

    train_datapool = _get_datapool_by_split("train")

    if remote_train:
        samples_by_split = {
            "train": [(sample, weight) for sample, weight in train_datapool],
        }
        return samples_by_split

    val_datapool = _get_datapool_by_split("val")
    test_datapool = _get_datapool_by_split("test")

    samples_by_split = {
        "train": [(sample, weight) for sample, weight in train_datapool],
        "val": [sample for sample, _ in val_datapool],
        "test": [sample for sample, _ in test_datapool]
    }
    return samples_by_split
