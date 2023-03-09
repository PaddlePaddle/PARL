from transformers import AutoTokenizer
from parl.utils import logger
from .reward_util import RougeRewardFunction
from .metric_util import MetricRegistry
from .data_pool import CNNDailyMail

def build_tokenizer(tokenizer_config):
    logger.info(f"loading tokenizer of [{tokenizer_config['model_name']}] model")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_config["model_name"])
    except Exception:
        logger.info(f"trying to use local_files to load tokenizer of [{tokenizer_config['model_name']}] model")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_config["model_name"], local_files_only=True)
    if tokenizer.pad_token is None and tokenizer_config.get("pad_token_as_eos_token", True):
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = tokenizer_config.get(
        "padding_side", "left")
    tokenizer.truncation_side = tokenizer_config.get(
        "truncation_side", "left")
    return tokenizer


def build_reward_fn(reward_config):
    logger.info(f"loading reward function: rouge")
    reward_fn = RougeRewardFunction(**reward_config.get("args", {}))
    return reward_fn


def build_metrics(metric_configs):
    metrics = [MetricRegistry.get(metric_config["id"], metric_config.get("args", {}))
               for metric_config in metric_configs]
    return metrics


def build_datapool(datapool_config, remote_train=False):
    def _get_datapool_by_split(split):
        kwargs = datapool_config.get("args", {})
        kwargs["split"] = split
        logger.info(f"loading split of dataset: {datapool_config['id']} -- {kwargs['split']}")
        dp_split = CNNDailyMail.prepare(**kwargs)
        logger.info(f"finish loading split of dataset: {datapool_config['id']} -- {kwargs['split']}")
        return dp_split

    train_datapool = _get_datapool_by_split("train")

    if remote_train:
        samples_by_split = {
            "train": [(sample, weight)
                      for sample, weight in train_datapool],
        }
        return samples_by_split

    val_datapool = _get_datapool_by_split("val")
    test_datapool = _get_datapool_by_split("test")

    samples_by_split = {
        "train": [(sample, weight)
                  for sample, weight in train_datapool],
        "val": [sample for sample, _ in val_datapool],
        "test": [sample for sample, _ in test_datapool]
    }
    return samples_by_split


