from datasets import load_metric


class RougeRewardFunction:
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
