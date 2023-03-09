from typing import Any, Dict, List

from tqdm import tqdm
from transformers import AutoTokenizer

from . import Sample
from parl.utils import logger


def get_batch(samples, batch_size):
    current_ix = 0
    n_samples = len(samples)
    while current_ix < n_samples:
        current_batch = samples[current_ix : current_ix + batch_size]
        yield current_batch
        current_ix += batch_size


def evaluate_on_samples(
    policy,
    tokenizer,
    samples,
    batch_size,
    max_prompt_length,
    metrics,
    epoch,
    split_name,
    dt_control_token = "",
    gen_kwargs = None,
):
    # generate text by batch
    all_generated_texts = []
    all_ref_texts = []
    all_prompt_texts = []
    all_meta_infos = []
    ###########CHANGE FOR DEBUG############
    tem = []
    for i in range(100):
        tem.append(samples[i])
    samples = tem
    ###########CHANGE FOR DEBUG############

    n_samples = len(samples)
    for batch in tqdm(list(get_batch(samples, batch_size)), desc="Evaluating"):
        batch_generated_texts = generate_text(
            policy, tokenizer, batch, max_prompt_length, dt_control_token, gen_kwargs
        )
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
    if metrics is not None:
        for metric in metrics:
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
        zip(samples, all_prompt_texts, all_generated_texts, all_ref_texts)
    ):
        sample_prediction = {
            "split_name": split_name,
            "sample_id": sample.id,
            "prompt_text": prompt_text,
            "generated_text": generated_text,
            "ref_text": "".join(
                [
                    f"<START-{ref_ix+1}>" + ref_text + f"<END-{ref_ix+1}>"
                    for ref_ix, ref_text in enumerate(ref_texts)
                ]
            ),
        }
        for metric_key, sample_scores in sample_scores_by_metric.items():
            sample_prediction[metric_key] = sample_scores[ix]
        sample_predictions_dict.append(sample_prediction)

    metrics_dict_ = {
        "epoch": epoch,
        "metrics": corpus_level_metrics
    }

    # logger
    logger.info(f"{split_name} metrics: {metrics_dict_}")


def generate_text(
    policy,
    tokenizer,
    samples,
    max_prompt_length,
    dt_control_token,
    gen_kwargs,
):
    prompt_texts = [
        dt_control_token + sample.prompt_or_input_text for sample in samples
    ]
    generated_texts = policy.sample(
        tokenizer, prompt_texts, max_prompt_length, gen_kwargs=gen_kwargs
    ).gen_texts
    return generated_texts
