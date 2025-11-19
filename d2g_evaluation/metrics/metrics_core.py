from dataclasses import asdict, dataclass
from enum import StrEnum, unique

from d2g_evaluation.text_preprocessing.text_preprocessing_core import TextPreprocessingResult
from d2g_evaluation.types import InputFormat


@unique
class MetricBackend(StrEnum):
    RAPIDFUZZ = "rapidfuzz"
    CYDIFFLIB = "cydifflib"
    OTHER = "other"


@unique
class ImplementedRapidFuzzMetrics(StrEnum):
    DAMERAU_LEVENSHTEIN = "damerau_levenshtein"
    INDEL = "indel"
    JARO = "jaro"
    JARO_WINKLER = "jaro_winkler"
    LEVENSHTEIN = "levenshtein"
    LONGEST_COMMON_SUBSEQUENCE = "longest_common_subsequence"
    OPTIMAL_STRING_ALIGNMENT = "optimal_string_alignment"
    PREFIX = "prefix"
    POSTFIX = "postfix"


@unique
class ImplementedCyDiffLibMetrics(StrEnum):
    RATCLIFF_OBERSHELP = "ratcliff_obershelp"


@unique
class ImplementedCustomMetrics(StrEnum):
    GREEDY_SEQUENTIAL_TOKEN_MATCHING = "greedy_sequential_token_matching"
    UNORDERED_TOKEN_MATCHING = "unordered_token_matching"
    LCS_TOKEN_MATCHING = "lcs_token_matching"
    RATCLIFF_OBERSHELP_TOKEN_MATCHING = "ratcliff_obershelp_token_matching"


@unique
class RapidFuzzOperation(StrEnum):
    DISTANCE = "distance"
    NORMALIZED_DISTANCE = "normalized_distance"
    SIMILARITY = "similarity"
    NORMALIZED_SIMILARITY = "normalized_similarity"


@unique
class CyDiffLibOperation(StrEnum):
    RATIO = "ratio"
    QUICK_RATIO = "quick_ratio"
    REAL_QUICK_RATIO = "real_quick_ratio"


@dataclass(slots=True)
class MetricResult:
    metric_name: str
    reference_length: int
    candidate_length: int
    is_symmetric: bool


@dataclass(slots=True)
class ScoreMetricResult(MetricResult):
    score: int | float
    is_symmetric_forced: bool = False

    def to_dict(self) -> dict[str, float | int | str | bool]:
        return asdict(self)


@dataclass(slots=True)
class FScoreMetricResult(MetricResult):
    true_positive: int
    false_positive: int
    false_negative: int

    is_symmetric_forced: bool = False

    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    f2: float = 0.0
    f05: float = 0.0

    def __post_init__(self) -> None:
        self._compute_metrics()
        self._validate_metrics()

    def _compute_metrics(self) -> None:
        """Compute precision, recall, and F-scores."""
        # handle edge case: both reference and candidate are empty
        if self.reference_length == 0 and self.candidate_length == 0:
            self.precision = 1.0
            self.recall = 1.0
        else:
            # compute precision and recall
            self.precision = (
                self.true_positive / (self.true_positive + self.false_positive)
                if (self.true_positive + self.false_positive) > 0
                else 0.0
            )
            self.recall = (
                self.true_positive / (self.true_positive + self.false_negative)
                if (self.true_positive + self.false_negative) > 0
                else 0.0
            )

        # compute F-scores
        self.f1 = self._f_beta(1.0)
        self.f2 = self._f_beta(2.0)
        self.f05 = self._f_beta(0.5)

    def _f_beta(self, beta: float) -> float:
        if (self.precision + self.recall) == 0:
            return 0.0
        beta_sq = beta**2
        return (1 + beta_sq) * (self.precision * self.recall) / ((beta_sq * self.precision) + self.recall)

    def _validate_metrics(self) -> None:
        assert 0 <= self.precision <= 1, f"Precision out of bounds: {self.precision}"
        assert 0 <= self.recall <= 1, f"Recall out of bounds: {self.recall}"
        assert 0 <= self.f1 <= 1, f"F1 out of bounds: {self.f1}"
        assert 0 <= self.f2 <= 1, f"F2 out of bounds: {self.f2}"
        assert 0 <= self.f05 <= 1, f"F05 out of bounds: {self.f05}"

    def to_dict(self) -> dict[str, float | int | str | bool]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class MetricConfig:
    name: ImplementedRapidFuzzMetrics | ImplementedCyDiffLibMetrics | ImplementedCustomMetrics
    backend: MetricBackend
    works_with: InputFormat
    output_type: type[ScoreMetricResult | FScoreMetricResult]  # accepts class types
    fully_symmetric: bool  # if True, metric(ref, cand) == metric(cand, ref)
    description: str
    # available_kwargs: tuple[str, ...] = ()


@dataclass(slots=True)
class MetricComputationResult:
    metric_result: ScoreMetricResult | FScoreMetricResult
    reference: TextPreprocessingResult
    candidate: TextPreprocessingResult

    def to_dict(self) -> dict[str, dict | str | int | float | bool]:
        return asdict(self)
