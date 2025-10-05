from enum import Enum

from d2g_evaluation.metrics.metrics_core import (
    ImplementedCyDiffLibMetrics,
    ImplementedRapidFuzzMetrics,
    MetricBackend,
    MetricConfig,
)
from d2g_evaluation.types import InputFormat


class ImplementedMetrics(Enum):
    def __init__(self, config: MetricConfig) -> None:
        self._config = config

    @property
    def config(self) -> MetricConfig:
        return self._config

    DAMERAU_LEVENSHTEIN = MetricConfig(
        name=ImplementedRapidFuzzMetrics.DAMERAU_LEVENSHTEIN,
        backend=MetricBackend.RAPIDFUZZ,
        works_with=InputFormat.STRING_AND_TOKEN,
        description="dummy description",
    )

    INDEL = MetricConfig(
        name=ImplementedRapidFuzzMetrics.INDEL,
        backend=MetricBackend.RAPIDFUZZ,
        works_with=InputFormat.STRING_AND_TOKEN,
        description="dummy description",
    )

    JARO = MetricConfig(
        name=ImplementedRapidFuzzMetrics.JARO,
        backend=MetricBackend.RAPIDFUZZ,
        works_with=InputFormat.STRING,
        description="dummy description",
    )

    JARO_WINKLER = MetricConfig(
        name=ImplementedRapidFuzzMetrics.JARO_WINKLER,
        backend=MetricBackend.RAPIDFUZZ,
        works_with=InputFormat.STRING,
        description="dummy description",
    )

    LEVENSHTEIN = MetricConfig(
        name=ImplementedRapidFuzzMetrics.LEVENSHTEIN,
        backend=MetricBackend.RAPIDFUZZ,
        works_with=InputFormat.STRING_AND_TOKEN,
        description="dummy description",
    )

    LONGEST_COMMON_SUBSEQUENCE = MetricConfig(
        name=ImplementedRapidFuzzMetrics.LONGEST_COMMON_SUBSEQUENCE,
        backend=MetricBackend.RAPIDFUZZ,
        works_with=InputFormat.STRING_AND_TOKEN,
        description="dummy description",
    )

    OPTIMAL_STRING_ALIGNMENT = MetricConfig(
        name=ImplementedRapidFuzzMetrics.OPTIMAL_STRING_ALIGNMENT,
        backend=MetricBackend.RAPIDFUZZ,
        works_with=InputFormat.STRING_AND_TOKEN,
        description="dummy description",
    )

    PREFIX = MetricConfig(
        name=ImplementedRapidFuzzMetrics.PREFIX,
        backend=MetricBackend.RAPIDFUZZ,
        works_with=InputFormat.STRING_AND_TOKEN,
        description="dummy description",
    )

    POSTFIX = MetricConfig(
        name=ImplementedRapidFuzzMetrics.POSTFIX,
        backend=MetricBackend.RAPIDFUZZ,
        works_with=InputFormat.STRING_AND_TOKEN,
        description="dummy description",
    )

    RATCLIFF_OBERSHELP = MetricConfig(
        name=ImplementedCyDiffLibMetrics.RATCLIFF_OBERSHELP,
        backend=MetricBackend.CYDIFFLIB,
        works_with=InputFormat.STRING_AND_TOKEN,
        description="dummy description",
    )
