# from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from enum import StrEnum, unique

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


# @unique
# class ImplementedCustomMetrics(StrEnum):
#     msg = "Custom metric not implemented yet."
#     raise NotImplementedError(msg)


@dataclass(slots=True)
class StringBasedMetricResult:
    score: float
    # reference: Sequence[Hashable] | str
    # candidate: Sequence[Hashable] | str
    # reference_length: int
    # candidate_length: int

    # def __post_init__(self) -> None:
    #     self.reference_length = len(self.reference)
    #     self.candidate_length = len(self.candidate)


@dataclass(slots=True)
class TokenBasedMetricResult:
    reference_tokens: int
    candidate_tokens: int
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float
    f2: float
    f05: float


@dataclass(slots=True, frozen=True)
class MetricConfig:
    name: str
    backend: MetricBackend
    works_with: InputFormat
    description: str
    # available_kwargs: tuple[str, ...] = ()
