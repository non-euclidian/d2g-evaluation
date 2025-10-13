import logging
from dataclasses import replace
from typing import Any, overload

from d2g_evaluation.metrics.metrics import CustomTokenMetricsWrapper, CyDiffLibWrapper, RapidFuzzWrapper
from d2g_evaluation.metrics.metrics_core import (
    CyDiffLibOperation,
    FScoreMetricResult,
    ImplementedCustomMetrics,
    ImplementedCyDiffLibMetrics,
    ImplementedRapidFuzzMetrics,
    MetricComputationResult,
    RapidFuzzOperation,
    ScoreMetricResult,
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

    def prepare_texts(  # noqa: PLR0913
        self,
        reference: str | list[str],
        candidate: str | list[str],
        string_preprocessing_method: ImplementedStringPreprocessing | None = None,
        tokenization_method: ImplementedTokenization | None = None,
        n: int = 3,
        *,
        already_preprocessed: bool = False,
    ) -> tuple[TextPreprocessingResult, TextPreprocessingResult]:
        reference_prepr = self.preprocessor.preprocess(
            text=reference,
            string_preprocessing_method=string_preprocessing_method,
            tokenization_method=tokenization_method,
            n=n,
            already_preprocessed=already_preprocessed,
        )
        candidate_prepr = self.preprocessor.preprocess(
            text=candidate,
            string_preprocessing_method=string_preprocessing_method,
            tokenization_method=tokenization_method,
            n=n,
            already_preprocessed=already_preprocessed,
        )
        return reference_prepr, candidate_prepr

    def _compute_metric_result(  # noqa: C901, PLR0912, PLR0915
        self,
        *,
        is_symmetric_forced: bool,
        wrapper: RapidFuzzWrapper | CyDiffLibWrapper | CustomTokenMetricsWrapper,
        reference_prepr: TextPreprocessingResult,
        candidate_prepr: TextPreprocessingResult,
        operation: RapidFuzzOperation | CyDiffLibOperation | None = None,
        **metric_kwargs: Any,  # noqa: ANN401
    ) -> MetricComputationResult:
        metric_result: ScoreMetricResult | FScoreMetricResult
        reverse_result: ScoreMetricResult | FScoreMetricResult

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
            if metric_result.is_symmetric is False and is_symmetric_forced is True:
                reverse_result = wrapper.calculate(
                    reference=candidate_prepr.result,
                    candidate=reference_prepr.result,
                    operation=operation,
                    **metric_kwargs,
                )
                avg_score = (metric_result.score + reverse_result.score) / 2
                avg_reference_length = int((metric_result.reference_length + reverse_result.reference_length) / 2)
                avg_candidate_length = int((metric_result.candidate_length + reverse_result.candidate_length) / 2)
                metric_result = replace(
                    metric_result,
                    score=avg_score,
                    reference_length=avg_reference_length,
                    candidate_length=avg_candidate_length,
                    is_symmetric=True,
                    is_symmetric_forced=is_symmetric_forced,
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

            if metric_result.is_symmetric is False and is_symmetric_forced is True:
                reverse_result = wrapper.calculate(
                    reference=candidate_prepr.result,
                    candidate=reference_prepr.result,
                    operation=operation,
                    **metric_kwargs,
                )

                avg_score = (metric_result.score + reverse_result.score) / 2
                avg_reference_length = int((metric_result.reference_length + reverse_result.reference_length) / 2)
                avg_candidate_length = int((metric_result.candidate_length + reverse_result.candidate_length) / 2)
                metric_result = replace(
                    metric_result,
                    score=avg_score,
                    reference_length=avg_reference_length,
                    candidate_length=avg_candidate_length,
                    is_symmetric=True,
                    is_symmetric_forced=is_symmetric_forced,
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

            metric_result = wrapper.calculate(reference=reference_prepr.result, candidate=candidate_prepr.result)
            if metric_result.is_symmetric is False and is_symmetric_forced is True:
                reverse_result = wrapper.calculate(reference=candidate_prepr.result, candidate=reference_prepr.result)

                f1_avg = (metric_result.f1 + reverse_result.f1) / 2
                f2_avg = (metric_result.f2 + reverse_result.f2) / 2
                f05_avg = (metric_result.f05 + reverse_result.f05) / 2
                precision_avg = (metric_result.precision + reverse_result.precision) / 2
                recall_avg = (metric_result.recall + reverse_result.recall) / 2

                avg_reference_length = int((metric_result.reference_length + reverse_result.reference_length) / 2)
                avg_candidate_length = int((metric_result.candidate_length + reverse_result.candidate_length) / 2)

                tp = metric_result.true_positive + reverse_result.true_positive
                fp = metric_result.false_positive + reverse_result.false_positive
                fn = metric_result.false_negative + reverse_result.false_negative

                metric_result = replace(
                    metric_result,
                    f1=f1_avg,
                    f2=f2_avg,
                    f05=f05_avg,
                    precision=precision_avg,
                    recall=recall_avg,
                    true_positive=tp,
                    false_positive=fp,
                    false_negative=fn,
                    reference_length=avg_reference_length,
                    candidate_length=avg_candidate_length,
                    is_symmetric=True,
                    is_symmetric_forced=is_symmetric_forced,
                )

        else:
            msg = f"Unsupported wrapper type: {type(wrapper)}"
            self.logger.error(msg)
            raise TypeError(msg)

        return MetricComputationResult(
            metric_result=metric_result,
            reference=reference_prepr,
            candidate=candidate_prepr,
        )

    def calculate(  # noqa: PLR0913
        self,
        *,
        reference: str | list[str],
        candidate: str | list[str],
        is_symmetric_forced: bool = True,
        metric_name: ImplementedRapidFuzzMetrics | ImplementedCyDiffLibMetrics | ImplementedCustomMetrics,
        operation: RapidFuzzOperation | CyDiffLibOperation | None = None,
        string_preprocessing_method: ImplementedStringPreprocessing | None = None,
        tokenization_method: ImplementedTokenization | None = None,
        n: int = 3,
        already_preprocessed: bool = False,
        **metric_kwargs: Any,  # noqa: ANN401
    ) -> MetricComputationResult:
        wrapper = self._get_wrapper(metric_name)

        reference_prepr, candidate_prepr = self.prepare_texts(
            reference=reference,
            candidate=candidate,
            string_preprocessing_method=string_preprocessing_method,
            tokenization_method=tokenization_method,
            n=n,
            already_preprocessed=already_preprocessed,
        )

        metric_computation_result = self._compute_metric_result(
            is_symmetric_forced=is_symmetric_forced,
            wrapper=wrapper,
            reference_prepr=reference_prepr,
            candidate_prepr=candidate_prepr,
            operation=operation,
            **metric_kwargs,
        )
        return metric_computation_result  # noqa: RET504
