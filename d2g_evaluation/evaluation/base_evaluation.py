from __future__ import annotations

import logging  # noqa: F401
from abc import ABC  # noqa: F401
from dataclasses import asdict, dataclass
from statistics import mean, median
from typing import Any, ClassVar  # noqa: F401

from d2g_evaluation.metrics.interface_metrics import InterfaceMetrics  # noqa: F401
from d2g_evaluation.metrics.metrics_core import FScoreMetricResult, MetricComputationResult, ScoreMetricResult


@dataclass(slots=True)
class PairwiseComparisonResult:
    computation: MetricComputationResult

    task_id: int | str | None = None
    reference_id: int | str | None = None
    candidate_id: int | str | None = None

    @property
    def metric_result(self) -> ScoreMetricResult | FScoreMetricResult:
        return self.computation.metric_result

    @property
    def metric_name(self) -> str:
        return self.metric_result.metric_name

    @property
    def is_reference_tokenized(self) -> bool:
        return self.computation.reference.is_tokenized

    @property
    def is_candidate_tokenized(self) -> bool:
        return self.computation.candidate.is_tokenized

    @property
    def reference_config_name(self) -> str:
        return self.computation.reference.config_name

    @property
    def candidate_config_name(self) -> str:
        return self.computation.candidate.config_name

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
    metric_name: str
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
        metric_name: str,
        task_id: int | str | None = None,
        pairwise_results: list[PairwiseComparisonResult] | None = None,
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


# class BaseEvaluation(ABC):
#     SEED: ClassVar[int] = 414242

#     def __init__(self) -> None:
#         self.logger = logging.getLogger(__name__)
#         self.interface_metrics = InterfaceMetrics()
