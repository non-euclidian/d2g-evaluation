import logging
from typing import Any, overload

from d2g_evaluation.metrics.metrics import CustomTokenMetricsWrapper, CyDiffLibWrapper, RapidFuzzWrapper
from d2g_evaluation.metrics.metrics_core import (
    CyDiffLibOperation,
    ImplementedCustomMetrics,
    ImplementedCyDiffLibMetrics,
    ImplementedRapidFuzzMetrics,
    MetricComputationResult,
    RapidFuzzOperation,
)
from d2g_evaluation.text_preprocessing.text_preprocessing import TextPreprocessor
from d2g_evaluation.text_preprocessing.text_preprocessing_core import (
    ImplementedStringPreprocessing,
    ImplementedTokenization,
    TextPreprocessingResult,
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

        # create and cache wrapper based on type
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

    def prepare_texts(
        self,
        reference: str,
        candidate: str,
        string_preprocessing_method: ImplementedStringPreprocessing | None = None,
        tokenization_method: ImplementedTokenization | None = None,
        n: int = 3,
    ) -> tuple[TextPreprocessingResult, TextPreprocessingResult]:
        reference_prepr = self.preprocessor.preprocess(
            text=reference,
            string_preprocessing_method=string_preprocessing_method,
            tokenization_method=tokenization_method,
            n=n,
        )
        candidate_prepr = self.preprocessor.preprocess(
            text=candidate,
            string_preprocessing_method=string_preprocessing_method,
            tokenization_method=tokenization_method,
            n=n,
        )
        return reference_prepr, candidate_prepr

    def calculate(  # noqa: C901, PLR0913
        self,
        reference: str | list[str],
        candidate: str | list[str],
        metric_name: ImplementedRapidFuzzMetrics | ImplementedCyDiffLibMetrics | ImplementedCustomMetrics,
        operation: RapidFuzzOperation | CyDiffLibOperation | None = None,
        *,
        already_preprocessed: bool = False,
        string_preprocessing_method: ImplementedStringPreprocessing | None = None,
        tokenization_method: ImplementedTokenization | None = None,
        n: int = 3,
        **metric_kwargs: Any,  # noqa: ANN401
    ) -> MetricComputationResult:
        wrapper = self._get_wrapper(metric_name)

        if already_preprocessed is True:
            msg = "Preprocessed reference and candidate are not yet supported."
            self.logger.error(msg)
            raise NotImplementedError(msg)

        if not isinstance(reference, str) or not isinstance(candidate, str):
            msg = "When `already_preprocessed` is `False`, both reference and candidate must be strings."
            self.logger.error(msg)
            raise TypeError(msg)

        reference_prepr, candidate_prepr = self.prepare_texts(
            reference=reference,
            candidate=candidate,
            string_preprocessing_method=string_preprocessing_method,
            tokenization_method=tokenization_method,
            n=n,
        )

        if isinstance(wrapper, RapidFuzzWrapper):
            if operation is None:
                operation = RapidFuzzOperation.NORMALIZED_SIMILARITY
            elif not isinstance(operation, RapidFuzzOperation):
                msg = f"Operation must be of type RapidFuzzOperation for RapidFuzz metrics, got {type(operation)}"
                self.logger.error(msg)
                raise TypeError(msg)

            metric_result = wrapper.calculate(
                reference=reference_prepr.result,
                candidate=candidate_prepr.result,
                operation=operation,
                **metric_kwargs,
            )
        elif isinstance(wrapper, CyDiffLibWrapper):
            if operation is None:
                operation = CyDiffLibOperation.RATIO
            elif not isinstance(operation, CyDiffLibOperation):
                msg = f"Operation must be of type CyDiffLibOperation for CyDiffLib metrics, got {type(operation)}"
                self.logger.error(msg)
                raise TypeError(msg)

            metric_result = wrapper.calculate(
                reference=reference_prepr.result,
                candidate=candidate_prepr.result,
                operation=operation,
                **metric_kwargs,
            )
        elif isinstance(wrapper, CustomTokenMetricsWrapper):
            if not reference_prepr.is_tokenized or not candidate_prepr.is_tokenized:
                msg = "For CustomTokenMetrics, both reference and candidate must be tokenized."
                self.logger.error(msg)
                raise ValueError(msg)

            if not isinstance(reference_prepr.result, list) or not isinstance(candidate_prepr.result, list):
                msg = "Tokenized reference and candidate must be of type list[str]."
                self.logger.error(msg)
                raise TypeError(msg)

            metric_result = wrapper.calculate(reference=reference_prepr.result, candidate=candidate_prepr.result)  # type: ignore
        else:
            msg = f"Unsupported wrapper type: {type(wrapper)}"
            self.logger.error(msg)
            raise TypeError(msg)

        return MetricComputationResult(
            metric_result=metric_result,
            reference=reference_prepr,
            candidate=candidate_prepr,
        )
