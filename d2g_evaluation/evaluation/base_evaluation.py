from __future__ import annotations

import itertools
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from statistics import mean, median
from typing import TYPE_CHECKING, Any, ClassVar

import numpy  # noqa: ICN001
import polars  # noqa: ICN001
from scipy.stats import bootstrap  # type: ignore

from d2g_evaluation.dataloader.base_dataloader import D2GDataLoader
from d2g_evaluation.metrics.interface_metrics import InterfaceMetrics
from d2g_evaluation.metrics.metrics_core import (
    FScoreMetricResult,
    ImplementedCustomMetrics,
    ImplementedCyDiffLibMetrics,
    ImplementedRapidFuzzMetrics,
    MetricComputationResult,
    ScoreMetricResult,
)

if TYPE_CHECKING:
    import pathlib

    import datasets  # type: ignore


@dataclass(slots=True)
class PairwiseEvaluationResult:
    metric_computation: MetricComputationResult | None = None

    task_id: int | str | None = None
    reference_id: int | str | None = None
    candidate_id: int | str | None = None

    @property
    def metric_result(self) -> ScoreMetricResult | FScoreMetricResult:
        if self.metric_computation is None:
            msg = "Cannot access metric_result on empty PairwiseComparisonResult"
            raise ValueError(msg)
        return self.metric_computation.metric_result

    @property
    def metric_name(self) -> str:
        if self.metric_computation is None:
            msg = "Cannot access metric_name on empty PairwiseComparisonResult"
            raise ValueError(msg)
        return self.metric_result.metric_name

    @property
    def is_reference_tokenized(self) -> bool:
        if self.metric_computation is None:
            msg = "Cannot access is_reference_tokenized on empty PairwiseComparisonResult"
            raise ValueError(msg)
        return self.metric_computation.reference.is_tokenized

    @property
    def is_candidate_tokenized(self) -> bool:
        if self.metric_computation is None:
            msg = "Cannot access is_candidate_tokenized on empty PairwiseComparisonResult"
            raise ValueError(msg)
        return self.metric_computation.candidate.is_tokenized

    @property
    def reference_config_name(self) -> str:
        if self.metric_computation is None:
            msg = "Cannot access reference_config_name on empty PairwiseComparisonResult"
            raise ValueError(msg)
        return self.metric_computation.reference.config_name

    @property
    def candidate_config_name(self) -> str:
        if self.metric_computation is None:
            msg = "Cannot access candidate_config_name on empty PairwiseComparisonResult"
            raise ValueError(msg)
        return self.metric_computation.candidate.config_name

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class DescriptiveStatistics:
    mean: float
    median: float
    min: float
    max: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_values(cls, values: list[float | int]) -> DescriptiveStatistics | None:
        if not values:
            return None
        return cls(
            mean=mean(values),
            median=median(values),
            min=min(values),
            max=max(values),
        )


@dataclass(slots=True)
class SampleEvaluationResult:
    metric_name: ImplementedRapidFuzzMetrics | ImplementedCyDiffLibMetrics | ImplementedCustomMetrics | str
    num_comparisons: int
    task_id: int | str | None = None

    config_name: str | None = None
    is_tokenized: bool | None = None

    # common fields
    is_symmetric: bool | None = None
    is_symmetric_forced: bool | None = None

    # fields for Score-based metrics
    score: DescriptiveStatistics | None = None

    # fields for F-Score based metrics
    precision: DescriptiveStatistics | None = None
    recall: DescriptiveStatistics | None = None
    f1: DescriptiveStatistics | None = None
    f2: DescriptiveStatistics | None = None
    f05: DescriptiveStatistics | None = None
    true_positive_sum: int | None = None
    false_positive_sum: int | None = None
    false_negative_sum: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_pairwise_results(
        cls,
        metric_name: str | ImplementedRapidFuzzMetrics | ImplementedCyDiffLibMetrics | ImplementedCustomMetrics,
        task_id: int | str | None = None,
        pairwise_results: list[PairwiseEvaluationResult] | None = None,
    ) -> SampleEvaluationResult:
        if pairwise_results is None or len(pairwise_results) == 0:
            return cls(
                metric_name=metric_name,
                num_comparisons=0,
                task_id=task_id,
            )

        num_comparisons = len(pairwise_results)
        first_result = pairwise_results[0]
        metric_name = first_result.metric_name

        # check for consistent metric names
        if not all(r.metric_name == metric_name for r in pairwise_results):
            msg = "Inconsistent metric names in pairwise results"
            raise ValueError(msg)

        assert first_result.reference_config_name == first_result.candidate_config_name, "Config names must be the same"
        assert first_result.is_reference_tokenized == first_result.is_candidate_tokenized, (
            "Tokenization must be the same"
        )

        # aggregate common properties
        base_kwargs: dict[str, Any] = {
            "metric_name": metric_name,
            "num_comparisons": num_comparisons,
            "task_id": first_result.task_id,
            "config_name": first_result.reference_config_name,
            "is_tokenized": first_result.is_reference_tokenized,
            "is_symmetric": first_result.metric_result.is_symmetric,
            "is_symmetric_forced": first_result.metric_result.is_symmetric_forced,
        }

        if isinstance(first_result.metric_result, ScoreMetricResult):
            score_values = [
                r.metric_result.score for r in pairwise_results if isinstance(r.metric_result, ScoreMetricResult)
            ]
            return cls(score=DescriptiveStatistics.from_values(score_values), **base_kwargs)

        if isinstance(first_result.metric_result, FScoreMetricResult):
            f_results: list[FScoreMetricResult] = [
                r.metric_result for r in pairwise_results if isinstance(r.metric_result, FScoreMetricResult)
            ]
            return cls(
                precision=DescriptiveStatistics.from_values([mr.precision for mr in f_results]),
                recall=DescriptiveStatistics.from_values([mr.recall for mr in f_results]),
                f1=DescriptiveStatistics.from_values([mr.f1 for mr in f_results]),
                f2=DescriptiveStatistics.from_values([mr.f2 for mr in f_results]),
                f05=DescriptiveStatistics.from_values([mr.f05 for mr in f_results]),
                true_positive_sum=sum(mr.true_positive for mr in f_results),
                false_positive_sum=sum(mr.false_positive for mr in f_results),
                false_negative_sum=sum(mr.false_negative for mr in f_results),
                **base_kwargs,
            )

        msg = f"Unsupported MetricResult type: {type(first_result.metric_result)}"
        raise ValueError(msg)


class BaseEvaluation(ABC):
    SEED: ClassVar[int] = 414242

    PIPELINE_NAME: ClassVar[str] = "base_evaluation_pipeline"
    INSUFFICIENT_ANNOTATIONS: ClassVar[int] = 9999  # to be overridden in subclasses

    # dataset output columns
    SAMPLE_RESULT_COLUMN: ClassVar[str] = "sample_result"
    PAIRWISE_RESULTS_COLUMN: ClassVar[str] = "pairwise_results"

    _FILTER_CRITERION_FIELD: ClassVar[str] = "base_eval_filter_field"  # to be overridden in subclasses

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.rng_generator = random.Random(self.SEED)

        self.interface_metrics = InterfaceMetrics()

    def generate_pairwise_combinations(self, items: list[Any]) -> list[tuple[Any, Any]]:
        return list(itertools.combinations(items, 2))

    def generate_one_to_many_pairs(self, main_item: Any, items: list[Any]) -> list[tuple[Any, Any]]:  # noqa: ANN401
        return list(zip(itertools.repeat(main_item), items))

    def sample_with_limit(self, items: list[Any], limit: int) -> list[Any]:
        if limit is None or limit <= 0 or limit >= len(items):
            return items
        return self.rng_generator.sample(items, limit)

    def load_hf_dataset_from_file(self, file_path: str | pathlib.Path) -> datasets.Dataset:
        """Load a dataset from a local file."""
        self.logger.info("Loading data from file: %s", file_path)
        data = D2GDataLoader.from_file(file_path=file_path).to_dataset(split="test")
        self.logger.info("Loaded %d records from %s", len(data), file_path)
        return data

    def _filter_annotations_with_no_text(self, annotations: list[dict]) -> list[dict]:
        """Remove annotations with no text (len_merged_spans == 0)."""
        self.logger.debug("Before filtering, number of annotations: %d", len(annotations))
        filtered_annotations = [ann for ann in annotations if ann[self._FILTER_CRITERION_FIELD] != 0]
        self.logger.debug("After filtering, number of annotations: %d", len(filtered_annotations))
        return filtered_annotations

    def _handle_insufficient_annotations(
        self,
        task_id: int | str | None,
        num_annotations: int,
        metric_name: ImplementedRapidFuzzMetrics | ImplementedCyDiffLibMetrics | ImplementedCustomMetrics | str,
    ) -> tuple[SampleEvaluationResult, list[PairwiseEvaluationResult]]:
        """Handle samples with insufficient valid annotations."""
        self.logger.warning(
            "Sample with (task_id: %s) has less than %d valid annotations after filtering. Skipping.",
            task_id,
            self.INSUFFICIENT_ANNOTATIONS,
        )
        sample_result = SampleEvaluationResult.from_pairwise_results(metric_name=metric_name, task_id=task_id)
        pairwise_results = [
            PairwiseEvaluationResult(
                metric_computation=None,
                task_id=task_id,
                reference_id=None,
                candidate_id=None,
            )
            for _ in range(num_annotations)
        ]
        return sample_result, pairwise_results

    def _build_result_dict(
        self, sample_result: SampleEvaluationResult, pairwise_results: list[PairwiseEvaluationResult]
    ) -> dict[str, dict | list[dict]]:
        """Convert evaluation results to dictionary format."""
        return {
            self.SAMPLE_RESULT_COLUMN: sample_result.to_dict(),
            self.PAIRWISE_RESULTS_COLUMN: [pr.to_dict() for pr in pairwise_results],
        }

    @abstractmethod
    def evaluate(self, *args: Any, **kwargs: Any) -> Any: ...  # noqa: ANN401

    def create_report(self, evaluated_dataset: datasets.Dataset, min_comparisons: int = 1) -> polars.DataFrame:
        dataframe = evaluated_dataset.to_polars()

        assert dataframe.height == evaluated_dataset.num_rows, (
            "Dataframe height does not match number of rows in the evaluated dataset."
        )
        assert dataframe.width == evaluated_dataset.num_columns, (
            "Dataframe width does not match number of columns in the evaluated dataset."
        )

        # unnest the sample_result column
        dataframe = dataframe.select(self.SAMPLE_RESULT_COLUMN).unnest(self.SAMPLE_RESULT_COLUMN)

        # remove columns that contains only null values (it's unused columns for metrics)
        dataframe = dataframe.select([col for col in dataframe.columns if not dataframe[col].is_null().all()])

        # capture metadata before filtering
        total_tasks_before = dataframe.height
        total_comparisons_before = dataframe.select(polars.col("num_comparisons").sum()).item()

        # get constant metadata fields (same across all rows)
        metadata_cols = ["config_name", "is_symmetric", "is_symmetric_forced", "is_tokenized", "metric_name"]
        metadata = dataframe.select(metadata_cols).head(1).to_dicts()[0]

        # include only samples with at least `min_comparisons` pairwise comparisons
        dataframe = dataframe.filter(polars.col("num_comparisons") >= min_comparisons)

        # capture metadata after filtering
        total_tasks_after = dataframe.height
        total_comparisons_after = dataframe.select(polars.col("num_comparisons").sum()).item()

        report_rows = []

        # 1) Score-based metrics (only mean)
        if "score" in dataframe.columns:
            only_score = dataframe.select(polars.col("score").struct.field("mean")).to_numpy().flatten()

            # Calculate mean of means
            mean_of_mean = only_score.mean()

            # Bootstrap confidence interval
            results = bootstrap(
                data=(only_score,),
                statistic=numpy.mean,
                n_resamples=10_000,
                confidence_level=0.95,
                method="BCa",
                alternative="two-sided",
                random_state=self.SEED,
            )

            report_rows.append(
                {
                    "metric": "score",
                    "mean": float(mean_of_mean),
                    "bootstrap_ci_low": float(results.confidence_interval.low),
                    "bootstrap_ci_high": float(results.confidence_interval.high),
                    "bootstrap_std_error": float(results.standard_error),
                }
            )

        # 2) F-score-based metrics (precision, recall, F1, F2, F0.5)
        fscore_metrics = ["precision", "recall", "f1", "f2", "f05"]

        for metric in fscore_metrics:
            if metric in dataframe.columns:
                only_metric = dataframe.select(polars.col(metric).struct.field("mean")).to_numpy().flatten()

                # Calculate mean of means
                mean_of_metric = only_metric.mean()

                # Bootstrap confidence interval
                results = bootstrap(
                    data=(only_metric,),
                    statistic=numpy.mean,
                    n_resamples=10_000,
                    confidence_level=0.95,
                    method="BCa",
                    alternative="two-sided",
                    random_state=self.SEED,
                )

                report_rows.append(
                    {
                        "metric": metric,
                        "mean": float(mean_of_metric),
                        "bootstrap_ci_low": float(results.confidence_interval.low),
                        "bootstrap_ci_high": float(results.confidence_interval.high),
                        "bootstrap_std_error": float(results.standard_error),
                    }
                )

        # Create polars DataFrame from report rows
        report_df = polars.DataFrame(report_rows)

        # Add metadata columns to the report
        report_df = report_df.with_columns(
            [
                polars.lit(metadata["config_name"]).alias("config_name"),
                polars.lit(metadata["is_symmetric"]).alias("is_symmetric"),
                polars.lit(metadata["is_symmetric_forced"]).alias("is_symmetric_forced"),
                polars.lit(metadata["is_tokenized"]).alias("is_tokenized"),
                polars.lit(metadata["metric_name"]).alias("metric_name"),
                polars.lit(total_tasks_before).alias("total_tasks_before_filter"),
                polars.lit(total_tasks_after).alias("total_tasks_after_filter"),
                polars.lit(total_comparisons_before).alias("total_comparisons_before_filter"),
                polars.lit(total_comparisons_after).alias("total_comparisons_after_filter"),
                polars.lit(min_comparisons).alias("min_comparisons_threshold"),
            ]
        )

        # Reorder columns for better readability
        report_df = report_df.select(
            [
                "metric_name",
                "config_name",
                "is_symmetric",
                "is_symmetric_forced",
                "is_tokenized",
                "metric",
                "mean",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "bootstrap_std_error",
                "total_tasks_before_filter",
                "total_tasks_after_filter",
                "total_comparisons_before_filter",
                "total_comparisons_after_filter",
                "min_comparisons_threshold",
            ]
        )

        return report_df  # noqa: RET504
