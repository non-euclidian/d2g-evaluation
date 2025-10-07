from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Hashable
from typing import Any, ClassVar

from cydifflib import SequenceMatcher  # type: ignore[import]
from rapidfuzz.distance import OSA, DamerauLevenshtein, Indel, Jaro, JaroWinkler, LCSseq, Levenshtein, Postfix, Prefix

from d2g_evaluation.metrics.metrics_core import (
    CyDiffLibOperation,
    FScoreMetricResult,
    ImplementedCustomMetrics,
    ImplementedCyDiffLibMetrics,
    ImplementedRapidFuzzMetrics,
    MetricBackend,
    MetricResult,
    RapidFuzzOperation,
)
from d2g_evaluation.metrics.metrics_registry import MetricRegistry


class BaseMetricWrapper(ABC):
    """Base class for all metric wrappers providing common functionality."""

    _BACKEND: ClassVar[MetricBackend]

    def __init__(
        self,
        metric_name: ImplementedRapidFuzzMetrics | ImplementedCyDiffLibMetrics | ImplementedCustomMetrics,
    ) -> None:
        self.metric_name = metric_name

    @property
    def metric_registry_config(self) -> MetricRegistry:
        for metric in MetricRegistry:
            if metric.config.backend == self._BACKEND and metric.config.name == self.metric_name:
                return metric
        msg = f"Metric {self.metric_name} not found in MetricRegistry."
        raise ValueError(msg)

    @abstractmethod
    def calculate(self, *args: Any, **kwargs: Any) -> MetricResult | FScoreMetricResult:  # noqa: ANN401
        """Calculate the metric. Must be implemented by subclasses."""
        ...

    def _validate_inputs_mixed(self, reference: str | list[str], candidate: str | list[str]) -> None:
        if type(reference) is not type(candidate):
            msg = f"Reference and candidate must be of the same type. Got {type(reference)} and {type(candidate)}."
            raise TypeError(msg)
        if not isinstance(reference, (str, list)):
            msg = f"Reference and candidate must be of type str or list. Got {type(reference)}."
            raise TypeError(msg)

        if isinstance(reference, list):
            self._ensure_list_of_str(items=reference)
            self._ensure_list_of_str(items=candidate)  # type: ignore[arg-type]

    def _validate_inputs_tokens(self, reference: list[str], candidate: list[str]) -> None:
        if type(reference) is not type(candidate):
            msg = f"Reference and candidate must be of the same type. Got {type(reference)} and {type(candidate)}."
            raise TypeError(msg)

        if not isinstance(reference, list):
            msg = f"Reference and candidate must be of type list. Got {type(reference)}."
            raise TypeError(msg)

        self._ensure_list_of_str(items=reference)
        self._ensure_list_of_str(items=candidate)

    def _ensure_list_of_str(self, items: list[Any]) -> None:
        if not all(isinstance(item, str) for item in items):
            msg = "All items in the list must be of type str."
            raise TypeError(msg)


class RapidFuzzWrapper(BaseMetricWrapper):
    _BACKEND: ClassVar[MetricBackend] = MetricBackend.RAPIDFUZZ
    _METRIC_MAP: ClassVar[dict[ImplementedRapidFuzzMetrics, Any]] = {
        ImplementedRapidFuzzMetrics.DAMERAU_LEVENSHTEIN: DamerauLevenshtein,
        ImplementedRapidFuzzMetrics.INDEL: Indel,
        ImplementedRapidFuzzMetrics.JARO: Jaro,
        ImplementedRapidFuzzMetrics.JARO_WINKLER: JaroWinkler,
        ImplementedRapidFuzzMetrics.LEVENSHTEIN: Levenshtein,
        ImplementedRapidFuzzMetrics.LONGEST_COMMON_SUBSEQUENCE: LCSseq,
        ImplementedRapidFuzzMetrics.OPTIMAL_STRING_ALIGNMENT: OSA,
        ImplementedRapidFuzzMetrics.PREFIX: Prefix,
        ImplementedRapidFuzzMetrics.POSTFIX: Postfix,
    }

    def __init__(self, metric_name: ImplementedRapidFuzzMetrics) -> None:
        if (metric_module := self._METRIC_MAP.get(metric_name)) is None:
            msg = f"Metric {metric_name} not implemented in RapidFuzzWrapper. Expected one of: {list(self._METRIC_MAP.keys())}"
            raise ValueError(msg)

        super().__init__(metric_name)
        self._metric_module = metric_module

    def calculate(
        self,
        reference: str | list[str],
        candidate: str | list[str],
        operation: RapidFuzzOperation = RapidFuzzOperation.NORMALIZED_SIMILARITY,
        **kwargs: Any,  # noqa: ANN401
    ) -> MetricResult:
        self._validate_inputs_mixed(reference=reference, candidate=candidate)
        operation_func = getattr(self._metric_module, operation.value)
        score: float | int = operation_func(s1=reference, s2=candidate, **kwargs)
        return MetricResult(
            metric_name=self.metric_name,
            reference_length=len(reference),
            candidate_length=len(candidate),
            score=score,
            is_symmetric=self.metric_registry_config.config.fully_symmetric,
        )


class CyDiffLibWrapper(BaseMetricWrapper):
    _BACKEND: ClassVar[MetricBackend] = MetricBackend.CYDIFFLIB
    _METRIC_MAP: ClassVar[dict[ImplementedCyDiffLibMetrics, Any]] = {
        ImplementedCyDiffLibMetrics.RATCLIFF_OBERSHELP: SequenceMatcher,
    }

    def __init__(self, metric_name: ImplementedCyDiffLibMetrics) -> None:
        if (metric_module := self._METRIC_MAP.get(metric_name)) is None:
            msg = f"Metric {metric_name} not implemented in CyDiffLibWrapper. Expected one of: {list(self._METRIC_MAP.keys())}"
            raise ValueError(msg)

        super().__init__(metric_name)
        self._metric_module = metric_module

    def calculate(
        self,
        reference: str | list[str],
        candidate: str | list[str],
        operation: CyDiffLibOperation = CyDiffLibOperation.RATIO,
        isjunk: None | Callable[[Hashable], bool] = None,
        *,
        autojunk: bool = False,
    ) -> MetricResult:
        self._validate_inputs_mixed(reference=reference, candidate=candidate)
        match self.metric_name:
            case ImplementedCyDiffLibMetrics.RATCLIFF_OBERSHELP:  # !!! Asymetric similarity metric !!!
                sm = SequenceMatcher(isjunk=isjunk, a=reference, b=candidate, autojunk=autojunk)
                operation_func = getattr(sm, operation.value)
                score = operation_func()
                return MetricResult(
                    metric_name=self.metric_name,
                    reference_length=len(reference),
                    candidate_length=len(candidate),
                    score=score,
                    is_symmetric=self.metric_registry_config.config.fully_symmetric,
                )
            case _:
                msg = f"Metric {self.metric_name} calculation not implemented."
                raise NotImplementedError(msg)


class CustomMetrics:
    @staticmethod
    def greedy_sequential_token_matching(reference: list[str], candidate: list[str]) -> FScoreMetricResult:
        match_count = 0
        i, j = 0, 0

        while i < len(reference) and j < len(candidate):
            if reference[i] == candidate[j]:
                match_count += 1
                i += 1
            j += 1

        true_positive = match_count
        false_positive = len(candidate) - match_count
        false_negative = len(reference) - match_count

        return FScoreMetricResult(
            metric_name=ImplementedCustomMetrics.GREEDY_SEQUENTIAL_TOKEN_MATCHING,
            reference_length=len(reference),
            candidate_length=len(candidate),
            is_symmetric=False,
            true_positive=true_positive,
            false_positive=false_positive,
            false_negative=false_negative,
        )

    @staticmethod
    def unordered_token_matching(reference: list[str], candidate: list[str]) -> FScoreMetricResult:
        candidate_counter = Counter(candidate)
        reference_counter = Counter(reference)

        true_positive = sum((candidate_counter & reference_counter).values())
        false_positive = len(candidate) - true_positive
        false_negative = len(reference) - true_positive
        return FScoreMetricResult(
            metric_name=ImplementedCustomMetrics.UNORDERED_TOKEN_MATCHING,
            reference_length=len(reference),
            candidate_length=len(candidate),
            is_symmetric=False,
            true_positive=true_positive,
            false_positive=false_positive,
            false_negative=false_negative,
        )

    @staticmethod
    def lcs_token_matching(reference: list[str], candidate: list[str]) -> FScoreMetricResult:
        true_positive = LCSseq.similarity(s1=reference, s2=candidate)
        false_positive = len(candidate) - true_positive
        false_negative = len(reference) - true_positive
        return FScoreMetricResult(
            metric_name=ImplementedCustomMetrics.LCS_TOKEN_MATCHING,
            reference_length=len(reference),
            candidate_length=len(candidate),
            is_symmetric=False,
            true_positive=true_positive,
            false_positive=false_positive,
            false_negative=false_negative,
        )


class CustomTokenMetricsWrapper(BaseMetricWrapper):
    _BACKEND: ClassVar[MetricBackend] = MetricBackend.OTHER
    _METRIC_MAP: ClassVar[dict[ImplementedCustomMetrics, Callable[[list[str], list[str]], FScoreMetricResult]]] = {
        ImplementedCustomMetrics.GREEDY_SEQUENTIAL_TOKEN_MATCHING: CustomMetrics.greedy_sequential_token_matching,
        ImplementedCustomMetrics.UNORDERED_TOKEN_MATCHING: CustomMetrics.unordered_token_matching,
        ImplementedCustomMetrics.LCS_TOKEN_MATCHING: CustomMetrics.lcs_token_matching,
    }

    def __init__(self, metric_name: ImplementedCustomMetrics) -> None:
        if (metric_func := self._METRIC_MAP.get(metric_name)) is None:
            msg = f"Metric {metric_name} not implemented in CustomTokenMetricsWrapper. Expected one of: {list(self._METRIC_MAP.keys())}"
            raise ValueError(msg)

        super().__init__(metric_name)
        self._metric_func = metric_func

    def calculate(self, reference: list[str], candidate: list[str]) -> FScoreMetricResult:
        self._validate_inputs_tokens(reference=reference, candidate=candidate)
        result = self._metric_func(reference, candidate)
        # update the symmetric field from the registry config
        result.is_symmetric = self.metric_registry_config.config.fully_symmetric

        return result
