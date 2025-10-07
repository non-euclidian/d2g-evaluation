import logging
from typing import overload

from d2g_evaluation.metrics.metrics import CustomTokenMetricsWrapper, CyDiffLibWrapper, RapidFuzzWrapper
from d2g_evaluation.metrics.metrics_core import (
    ImplementedCustomMetrics,
    ImplementedCyDiffLibMetrics,
    ImplementedRapidFuzzMetrics,
)
from d2g_evaluation.text_preprocessing.text_preprocessing import TextPreprocessor
from d2g_evaluation.text_preprocessing.text_preprocessing_core import (
    ImplementedStringPreprocessing,  # noqa: F401
    ImplementedTokenization,  # noqa: F401
    TextPreprocessingResult,  # noqa: F401
)


class InterfaceMetrics:
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.preprocessor = TextPreprocessor()
        self._wrapper_cache: dict[
            ImplementedRapidFuzzMetrics | ImplementedCyDiffLibMetrics | ImplementedCustomMetrics,
            RapidFuzzWrapper | CyDiffLibWrapper | CustomTokenMetricsWrapper,
        ] = {}

    @overload
    def _get_wrapper(self, metric_name: ImplementedRapidFuzzMetrics) -> RapidFuzzWrapper: ...
    @overload
    def _get_wrapper(self, metric_name: ImplementedCyDiffLibMetrics) -> CyDiffLibWrapper: ...
    @overload
    def _get_wrapper(self, metric_name: ImplementedCustomMetrics) -> CustomTokenMetricsWrapper: ...

    def _get_wrapper(self, metric_name):
        """Get appropriate wrapper based on metric type."""

        if metric_name in self._wrapper_cache:
            return self._wrapper_cache[metric_name]

        # Create and cache wrapper based on type
        if isinstance(metric_name, ImplementedRapidFuzzMetrics):
            wrapper = RapidFuzzWrapper(metric_name=metric_name)
        elif isinstance(metric_name, ImplementedCyDiffLibMetrics):
            wrapper = CyDiffLibWrapper(metric_name=metric_name)
        elif isinstance(metric_name, ImplementedCustomMetrics):
            wrapper = CustomTokenMetricsWrapper(metric_name=metric_name)
        else:
            msg = f"Unknown metric type: {type(metric_name)}"
            self.logger.error(msg)
            raise TypeError(msg)

        self._wrapper_cache[metric_name] = wrapper
        return wrapper

    # def calculate(
    #     self,
    #     reference: str | list[str],
    #     candidate: str | list[str],
    #     *,

    #     string_preprocessing_method: ImplementedStringPreprocessing | None = None,
    #     tokenization_method: ImplementedTokenization | None = None,
    #     n: int = 3,
    # ) -> None:
