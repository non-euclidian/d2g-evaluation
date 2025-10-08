import logging
from typing import ClassVar

import polars  # noqa: ICN001

from d2g_evaluation.metrics.interface_metrics import InterfaceMetrics
from d2g_evaluation.metrics.metrics_registry import MetricRegistry
from d2g_evaluation.text_preprocessing.text_preprocessing_core import (
    ImplementedStringPreprocessing,
    ImplementedTokenization,
)
from d2g_evaluation.types import InputFormat

polars.Config.set_tbl_rows(-1)  # Show all rows
polars.Config.set_tbl_cols(-1)  # Show all columns
polars.Config.set_fmt_str_lengths(100)  # Set max string length for display


class MetricsPlayground:
    LIMITED_N_VALUES: ClassVar[list[int]] = [1, 2, 3, 4, 5]
    LIMITED_STRING_PREPROCESSING_METHODS: ClassVar[list[ImplementedStringPreprocessing]] = [
        ImplementedStringPreprocessing.NORMALIZE_STRING,
        # ImplementedStringPreprocessing.REMOVE_WHITESPACES,
    ]

    STRING_PREPROCESSING_FOR_TOKENIZATION: ClassVar[list[ImplementedStringPreprocessing]] = [
        ImplementedStringPreprocessing.NORMALIZE_STRING
    ]

    LIMITED_TOKENIZATION_METHODS: ClassVar[list[ImplementedTokenization]] = [
        ImplementedTokenization.CHAR_NGRAMS,
        # ImplementedTokenization.NCHARS,
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

        samples = []

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
                    samples.append(
                        {
                            "metric": result.metric_result.metric_name,
                            "score": round(result.metric_result.score, round_val),
                            "prepr": result.candidate.config_name,
                            "ref_len": result.metric_result.reference_length,
                            "cand_len": result.metric_result.candidate_length,
                            # "reference": result.reference.result,
                            # "candidate": result.candidate.result,
                            "is_symmetric": result.metric_result.is_symmetric,
                            "is_symmetric_forced": result.metric_result.is_symmetric_forced,
                        }
                    )

                else:
                    msg = f"Metric {metric.config.name} does not have 'score' attribute."
                    self.logger.error(msg)
                    raise AttributeError(msg)

        df = polars.DataFrame(samples)
        df = df.sort(by=["metric", "prepr"])
        print(df)

    def demo_all_token_metrics(
        self,
        reference: str,
        candidate: str,
        round_val: int = 2,
    ) -> None:
        metrics_that_work_with = self.all_metrics_by_formats(formats=[InputFormat.TOKEN, InputFormat.STRING_AND_TOKEN])

        score_samples = []
        f1_samples = []

        for metric in metrics_that_work_with:
            for str_prepr in self.STRING_PREPROCESSING_FOR_TOKENIZATION:
                for tokenization in self.LIMITED_TOKENIZATION_METHODS:
                    for n in self.LIMITED_N_VALUES:
                        result = self.interface_metrics.calculate(
                            reference=reference,
                            candidate=candidate,
                            metric_name=metric.config.name,
                            string_preprocessing_method=str_prepr,
                            tokenization_method=tokenization,
                            n=n,
                        )
                        if hasattr(result.metric_result, "score"):
                            score_samples.append(
                                {
                                    "metric": result.metric_result.metric_name,
                                    "score": round(result.metric_result.score, round_val),
                                    "prepr": result.candidate.config_name,
                                    "ref_len": result.metric_result.reference_length,
                                    "cand_len": result.metric_result.candidate_length,
                                    # "reference": result.reference.result,
                                    # "candidate": result.candidate.result,
                                    "is_symmetric": result.metric_result.is_symmetric,
                                    "is_symmetric_forced": result.metric_result.is_symmetric_forced,
                                }
                            )
                        elif hasattr(result.metric_result, "f1"):
                            f1_samples.append(
                                {
                                    "metric": result.metric_result.metric_name,
                                    "f1": round(result.metric_result.f1, round_val),
                                    "f2": round(result.metric_result.f2, round_val),
                                    "f05": round(result.metric_result.f05, round_val),
                                    "p": round(result.metric_result.precision, round_val),
                                    "r": round(result.metric_result.recall, round_val),
                                    "tp": result.metric_result.true_positive,
                                    "fp": result.metric_result.false_positive,
                                    "fn": result.metric_result.false_negative,
                                    "prepr": result.candidate.config_name,
                                    "ref_len": result.metric_result.reference_length,
                                    "cand_len": result.metric_result.candidate_length,
                                    # "reference": result.reference.result,
                                    # "candidate": result.candidate.result,
                                    "is_symmetric": result.metric_result.is_symmetric,
                                    "is_symmetric_forced": result.metric_result.is_symmetric_forced,
                                }
                            )

                        else:
                            msg = f"Metric {metric.config.name} does not have 'score' or 'f1' attribute."
                            self.logger.error(msg)
                            raise AttributeError(msg)

        df_scores = polars.DataFrame(score_samples)
        df_scores = df_scores.sort(by=["metric", "prepr"])
        print(df_scores)

        print("\n---\n")
        df_f1 = polars.DataFrame(f1_samples)
        df_f1 = df_f1.sort(by=["metric", "prepr"])
        print(df_f1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    playground = MetricsPlayground()

    ref_text = "This is a sample text! Now with more text."
    cand_text = "Have thsi. This is a sample text!"

    playground.demo_all_string_metrics(
        reference=ref_text,
        candidate=cand_text,
        round_val=3,
    )

    playground.demo_all_token_metrics(
        reference=ref_text,
        candidate=cand_text,
        round_val=3,
    )
