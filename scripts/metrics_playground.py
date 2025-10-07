import logging
from typing import ClassVar

from d2g_evaluation.metrics.interface_metrics import InterfaceMetrics
from d2g_evaluation.metrics.metrics_registry import MetricRegistry
from d2g_evaluation.text_preprocessing.text_preprocessing_core import (
    ImplementedStringPreprocessing,
    ImplementedTokenization,
)
from d2g_evaluation.types import InputFormat


class MetricsPlayground:
    LIMITED_N_VALUES: ClassVar[list[int]] = [1, 2, 3, 4, 5]
    LIMITED_STRING_PREPROCESSING_METHODS: ClassVar[list[ImplementedStringPreprocessing]] = [
        ImplementedStringPreprocessing.NORMALIZE_STRING,
        ImplementedStringPreprocessing.REMOVE_WHITESPACES,
    ]

    STRING_PREPROCESSING_FOR_TOKENIZATION: ClassVar[list[ImplementedStringPreprocessing]] = [
        ImplementedStringPreprocessing.NORMALIZE_STRING
    ]

    LIMITED_TOKENIZATION_METHODS: ClassVar[list[ImplementedTokenization]] = [
        ImplementedTokenization.CHAR_NGRAMS,
        ImplementedTokenization.NCHARS,
    ]

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.interface_metrics = InterfaceMetrics()

    def all_metrics_by_formats(self, formats: list[InputFormat]) -> list[MetricRegistry]:
        self.logger.info("Finding all metrics that work with formats: %s", formats)

        list_of_metrics = []
        for metric in MetricRegistry:
            if metric.config.works_with in formats:
                list_of_metrics.append(metric)  # noqa: PERF401

        self.logger.info("Found %d metrics", len(list_of_metrics))

        only_names = [m.name for m in list_of_metrics]
        self.logger.info("Metrics: %s", only_names)

        return list_of_metrics

    def demo_all_string_metrics(
        self,
        reference: str,
        candidate: str,
        round_val: int = 2,
    ) -> None:
        metrics_that_work_with = self.all_metrics_by_formats(formats=[InputFormat.STRING, InputFormat.STRING_AND_TOKEN])

        for metric in metrics_that_work_with:
            for str_prepr in self.LIMITED_STRING_PREPROCESSING_METHODS:
                result = self.interface_metrics.calculate(
                    reference=reference,
                    candidate=candidate,
                    metric_name=metric.config.name,
                    string_preprocessing_method=str_prepr,
                    tokenization_method=None,
                )
                if hasattr(result.metric_result, "score"):
                    print(round(result.metric_result.score, round_val))
                else:
                    msg = f"Metric {metric.config.name} does not have 'score' attribute."
                    self.logger.error(msg)
                    raise AttributeError(msg)

    def demo_all_token_metrics(
        self,
        reference: str,
        candidate: str,
        round_val: int = 2,
    ) -> None:
        metrics_that_work_with = self.all_metrics_by_formats(formats=[InputFormat.TOKEN, InputFormat.STRING_AND_TOKEN])

        for metric in metrics_that_work_with:
            for str_prepr in self.STRING_PREPROCESSING_FOR_TOKENIZATION:
                for tokenization in self.LIMITED_TOKENIZATION_METHODS:
                    result = self.interface_metrics.calculate(
                        reference=reference,
                        candidate=candidate,
                        metric_name=metric.config.name,
                        string_preprocessing_method=str_prepr,
                        tokenization_method=tokenization,
                    )
                    if hasattr(result.metric_result, "score"):
                        print(round(result.metric_result.score, round_val))
                    else:
                        print(result.metric_result.f1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    playground = MetricsPlayground()

    ref_text = "This is a sample reference text."
    cand_text = "This is a sample candidate text!"

    playground.demo_all_string_metrics(
        reference=ref_text,
        candidate=cand_text,
        round_val=3,
    )

    print("\n---\n")

    playground.demo_all_token_metrics(
        reference=ref_text,
        candidate=cand_text,
        round_val=3,
    )
