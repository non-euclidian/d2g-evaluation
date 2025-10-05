# from collections.abc import Hashable, Sequence
from typing import Any

import rapidfuzz  # type: ignore[import]

from d2g_evaluation.metrics.metrics_core import StringBasedMetricResult, TokenBasedMetricResult


class RapidFuzzWrapper:
    def __init__(self, rapidfuzz_distance: rapidfuzz.distance) -> None:
        self._module = rapidfuzz_distance

    def normalized_similarity(
        self,
        reference: str | list[str],
        candidate: str | list[str],
        **kwargs: Any,  # noqa: ANN401
    ) -> StringBasedMetricResult:
        score = self._module.normalized_similarity(s1=reference, s2=candidate, **kwargs)

        return StringBasedMetricResult(
            score=score,
        )


class CyDiffLibWrapper:
    def __init__(self, cydifflib_module: Any) -> None:  # noqa: ANN401
        self._module = cydifflib_module

    def ratio(self, reference: str | list[str], candidate: str | list[str], **kwargs: Any) -> float:  # noqa: ANN401
        return self._module.ratio(reference, candidate, **kwargs)


class TokenMetricWrapper:
    def __init__(self, token_metric_function: Any) -> None:  # noqa: ANN401
        self._function = token_metric_function

    def calculate(self, reference: list[str], candidate: list[str], **kwargs: Any) -> TokenBasedMetricResult:  # noqa: ANN401
        return self._function(reference, candidate, **kwargs)
