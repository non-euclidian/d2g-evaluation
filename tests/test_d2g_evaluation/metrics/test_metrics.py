import pytest

from d2g_evaluation.metrics.metrics import CustomTokenMetricsWrapper, CyDiffLibWrapper, RapidFuzzWrapper
from d2g_evaluation.metrics.metrics_core import (
    CyDiffLibOperation,
    FScoreMetricResult,
    ImplementedCustomMetrics,
    ImplementedCyDiffLibMetrics,
    ImplementedRapidFuzzMetrics,
    MetricConfig,
    RapidFuzzOperation,
    ScoreMetricResult,
)
from d2g_evaluation.metrics.metrics_registry import MetricRegistry
from tests.test_d2g_evaluation.metrics.conftest import (
    TEST_INVALID_INPUT_CUSTOM_TOKEN,
    TEST_INVALID_INPUT_CYDIFFLIB,
    TEST_INVALID_INPUT_RAPIDFUZZ,
    TEST_VALID_INPUT_SAME_STRING,
    TEST_VALID_INPUT_SAME_TOKEN,
    TEST_VALID_INPUT_STRING,
    TEST_VALID_INPUT_TOKEN,
    CalculateGenericTestCase,
    CustomTokenInvalidInputTestCase,
    CyDiffLibInvalidInputTestCase,
    RapidFuzzInvalidInputTestCase,
)


class TestRapidFuzzWrapper:
    @pytest.mark.parametrize("metric_name", list(ImplementedRapidFuzzMetrics))
    def test_init(self, metric_name: ImplementedRapidFuzzMetrics) -> None:
        wrapper = RapidFuzzWrapper(metric_name=metric_name)
        assert wrapper.metric_name == metric_name
        assert wrapper._metric_module is not None
        assert wrapper.metric_registry_config.config.name == metric_name

    def test_init_invalid_metric_name(self) -> None:
        with pytest.raises((ValueError, KeyError)):
            RapidFuzzWrapper(metric_name="invalid_metric")  # type: ignore

    @pytest.mark.parametrize("metric_name", list(ImplementedRapidFuzzMetrics))
    def test_metric_registry_config(self, metric_name: ImplementedRapidFuzzMetrics) -> None:
        wrapper = RapidFuzzWrapper(metric_name=metric_name)
        wrapper_config = wrapper.metric_registry_config

        assert isinstance(wrapper_config, MetricRegistry)
        assert isinstance(wrapper_config.config, MetricConfig)
        assert wrapper_config.config.name == metric_name
        assert wrapper_config.config.backend.name == "RAPIDFUZZ"

    @pytest.mark.parametrize("metric_name", list(ImplementedRapidFuzzMetrics))
    @pytest.mark.parametrize("operation", list(RapidFuzzOperation))
    @pytest.mark.parametrize("test_case", TEST_VALID_INPUT_STRING + TEST_VALID_INPUT_TOKEN, ids=lambda tc: tc.test_name)
    def test_calculate(
        self,
        metric_name: ImplementedRapidFuzzMetrics,
        operation: RapidFuzzOperation,
        test_case: CalculateGenericTestCase,
    ) -> None:
        wrapper = RapidFuzzWrapper(metric_name=metric_name)
        result = wrapper.calculate(reference=test_case.reference, candidate=test_case.candidate, operation=operation)
        assert isinstance(result, ScoreMetricResult)
        assert result.metric_name == metric_name
        assert result.reference_length == len(test_case.reference)
        assert result.candidate_length == len(test_case.candidate)
        assert isinstance(result.score, (int, float))

    @pytest.mark.parametrize("metric_name", list(ImplementedRapidFuzzMetrics))
    @pytest.mark.parametrize("operation", list(RapidFuzzOperation))
    @pytest.mark.parametrize(
        "test_case", TEST_VALID_INPUT_SAME_STRING + TEST_VALID_INPUT_SAME_TOKEN, ids=lambda tc: tc.test_name
    )
    def test_calculate_same_strings(
        self,
        metric_name: ImplementedRapidFuzzMetrics,
        operation: RapidFuzzOperation,
        test_case: CalculateGenericTestCase,
    ) -> None:
        wrapper = RapidFuzzWrapper(metric_name=metric_name)
        result = wrapper.calculate(reference=test_case.reference, candidate=test_case.candidate, operation=operation)
        assert isinstance(result, ScoreMetricResult)
        assert result.metric_name == metric_name
        assert result.reference_length == len(test_case.reference)
        assert result.candidate_length == len(test_case.candidate)

        if operation == RapidFuzzOperation.NORMALIZED_DISTANCE:
            assert result.score == 0.0  # normalized distance should be 0 for identical strings
        elif operation == RapidFuzzOperation.DISTANCE:
            assert result.score == 0  # raw distance should be 0 for identical strings
        elif operation == RapidFuzzOperation.NORMALIZED_SIMILARITY:
            assert result.score == 1.0  # normalized similarity should be 1.0 for identical strings
        else:  # SIMILARITY operations
            assert isinstance(result.score, (int, float))  # just check it's a number
            assert result.score >= 0  # should be positive for identical strings

    @pytest.mark.parametrize("metric_name", list(ImplementedRapidFuzzMetrics))
    @pytest.mark.parametrize("operation", list(RapidFuzzOperation))
    @pytest.mark.parametrize("test_case", TEST_INVALID_INPUT_RAPIDFUZZ, ids=lambda tc: tc.test_name)
    def test_calculate_invalid_input_rapidfuzz(
        self,
        metric_name: ImplementedRapidFuzzMetrics,
        operation: RapidFuzzOperation,
        test_case: RapidFuzzInvalidInputTestCase,
    ) -> None:
        wrapper = RapidFuzzWrapper(metric_name=metric_name)

        with pytest.raises(test_case.expected_exception):
            wrapper.calculate(reference=test_case.reference, candidate=test_case.candidate, operation=operation)  # type: ignore

    @pytest.mark.parametrize("metric_name", list(ImplementedRapidFuzzMetrics))
    @pytest.mark.parametrize("operation", list(RapidFuzzOperation))
    @pytest.mark.parametrize(
        "test_case",
        TEST_VALID_INPUT_STRING + TEST_VALID_INPUT_SAME_STRING + TEST_VALID_INPUT_TOKEN + TEST_VALID_INPUT_SAME_TOKEN,
        ids=lambda tc: tc.test_name,
    )
    def test_calculate_bidirectional(
        self,
        metric_name: ImplementedRapidFuzzMetrics,
        operation: RapidFuzzOperation,
        test_case: CalculateGenericTestCase,
    ) -> None:
        wrapper = RapidFuzzWrapper(metric_name=metric_name)

        result1 = wrapper.calculate(reference=test_case.reference, candidate=test_case.candidate, operation=operation)
        result2 = wrapper.calculate(reference=test_case.candidate, candidate=test_case.reference, operation=operation)

        assert isinstance(result1, ScoreMetricResult)
        assert isinstance(result2, ScoreMetricResult)
        assert result1.metric_name == result2.metric_name == metric_name
        assert result1.reference_length == len(test_case.reference)
        assert result1.candidate_length == len(test_case.candidate)
        assert result2.reference_length == len(test_case.candidate)
        assert result2.candidate_length == len(test_case.reference)

        assert result1.score == result2.score


class TestCyDiffLibWrapper:
    @pytest.mark.parametrize("metric_name", list(ImplementedCyDiffLibMetrics))
    def test_init(self, metric_name: ImplementedCyDiffLibMetrics) -> None:
        wrapper = CyDiffLibWrapper(metric_name=metric_name)
        assert wrapper.metric_name == metric_name
        assert wrapper._metric_module is not None
        assert wrapper.metric_registry_config.config.name == metric_name

    def test_init_invalid_metric_name(self) -> None:
        with pytest.raises((ValueError, KeyError)):
            CyDiffLibWrapper(metric_name="invalid_metric")  # type: ignore

    @pytest.mark.parametrize("metric_name", list(ImplementedCyDiffLibMetrics))
    def test_metric_registry_config(self, metric_name: ImplementedCyDiffLibMetrics) -> None:
        wrapper = CyDiffLibWrapper(metric_name=metric_name)
        wrapper_config = wrapper.metric_registry_config

        assert isinstance(wrapper_config, MetricRegistry)
        assert isinstance(wrapper_config.config, MetricConfig)
        assert wrapper_config.config.name == metric_name
        assert wrapper_config.config.backend.name == "CYDIFFLIB"

    @pytest.mark.parametrize("metric_name", list(ImplementedCyDiffLibMetrics))
    @pytest.mark.parametrize("operation", list(CyDiffLibOperation))
    @pytest.mark.parametrize(
        "test_case", TEST_VALID_INPUT_STRING + TEST_VALID_INPUT_SAME_TOKEN, ids=lambda tc: tc.test_name
    )
    def test_calculate(
        self,
        metric_name: ImplementedCyDiffLibMetrics,
        operation: CyDiffLibOperation,
        test_case: CalculateGenericTestCase,
    ) -> None:
        wrapper = CyDiffLibWrapper(metric_name=metric_name)
        result = wrapper.calculate(reference=test_case.reference, candidate=test_case.candidate, operation=operation)
        assert isinstance(result, ScoreMetricResult)
        assert result.metric_name == metric_name
        assert result.reference_length == len(test_case.reference)
        assert result.candidate_length == len(test_case.candidate)
        assert isinstance(result.score, float)

    @pytest.mark.parametrize("metric_name", list(ImplementedCyDiffLibMetrics))
    @pytest.mark.parametrize("operation", list(CyDiffLibOperation))
    @pytest.mark.parametrize(
        "test_case", TEST_VALID_INPUT_SAME_STRING + TEST_VALID_INPUT_SAME_TOKEN, ids=lambda tc: tc.test_name
    )
    def test_calculate_same_strings(
        self,
        metric_name: ImplementedCyDiffLibMetrics,
        operation: CyDiffLibOperation,
        test_case: CalculateGenericTestCase,
    ) -> None:
        wrapper = CyDiffLibWrapper(metric_name=metric_name)
        result = wrapper.calculate(reference=test_case.reference, candidate=test_case.candidate, operation=operation)
        assert isinstance(result, ScoreMetricResult)
        assert result.metric_name == metric_name
        assert result.reference_length == len(test_case.reference)
        assert result.candidate_length == len(test_case.candidate)
        assert result.score == 1.0

    @pytest.mark.parametrize("metric_name", list(ImplementedCyDiffLibMetrics))
    @pytest.mark.parametrize("operation", list(CyDiffLibOperation))
    @pytest.mark.parametrize("test_case", TEST_INVALID_INPUT_CYDIFFLIB, ids=lambda tc: tc.test_name)
    def test_calculate_invalid_input_cydifflib(
        self,
        metric_name: ImplementedCyDiffLibMetrics,
        operation: CyDiffLibOperation,
        test_case: CyDiffLibInvalidInputTestCase,
    ) -> None:
        wrapper = CyDiffLibWrapper(metric_name=metric_name)

        with pytest.raises(test_case.expected_exception):
            wrapper.calculate(reference=test_case.reference, candidate=test_case.candidate, operation=operation)  # type: ignore

    @pytest.mark.parametrize("metric_name", list(ImplementedCyDiffLibMetrics))
    @pytest.mark.parametrize("operation", list(CyDiffLibOperation))
    @pytest.mark.parametrize(
        "test_case",
        TEST_VALID_INPUT_STRING + TEST_VALID_INPUT_SAME_STRING + TEST_VALID_INPUT_TOKEN + TEST_VALID_INPUT_SAME_TOKEN,
        ids=lambda tc: tc.test_name,
    )
    def test_calculate_bidirectional(
        self,
        metric_name: ImplementedCyDiffLibMetrics,
        operation: CyDiffLibOperation,
        test_case: CalculateGenericTestCase,
    ) -> None:
        wrapper = CyDiffLibWrapper(metric_name=metric_name)

        result1 = wrapper.calculate(reference=test_case.reference, candidate=test_case.candidate, operation=operation)
        result2 = wrapper.calculate(reference=test_case.candidate, candidate=test_case.reference, operation=operation)

        assert isinstance(result1, ScoreMetricResult)
        assert isinstance(result2, ScoreMetricResult)
        assert result1.metric_name == result2.metric_name == metric_name
        assert result1.reference_length == len(test_case.reference)
        assert result1.candidate_length == len(test_case.candidate)
        assert result2.reference_length == len(test_case.candidate)
        assert result2.candidate_length == len(test_case.reference)

        if metric_name == ImplementedCyDiffLibMetrics.RATCLIFF_OBERSHELP:  # not bidirectional
            assert isinstance(result1.score, float)
            assert isinstance(result2.score, float)
            assert 0 <= result1.score <= 1.0
            assert 0 <= result2.score <= 1.0


class TestCustomTokenMetricsWrapper:
    @pytest.mark.parametrize("metric_name", list(ImplementedCustomMetrics))
    def test_init(self, metric_name: ImplementedCustomMetrics) -> None:
        wrapper = CustomTokenMetricsWrapper(metric_name=metric_name)
        assert wrapper.metric_name == metric_name
        assert wrapper.metric_registry_config.config.name == metric_name

    def test_init_invalid_metric_name(self) -> None:
        with pytest.raises((ValueError, KeyError)):
            CustomTokenMetricsWrapper(metric_name="invalid_metric")  # type: ignore

    @pytest.mark.parametrize("metric_name", list(ImplementedCustomMetrics))
    def test_metric_registry_config(self, metric_name: ImplementedCustomMetrics) -> None:
        wrapper = CustomTokenMetricsWrapper(metric_name=metric_name)
        wrapper_config = wrapper.metric_registry_config

        assert isinstance(wrapper_config, MetricRegistry)
        assert isinstance(wrapper_config.config, MetricConfig)
        assert wrapper_config.config.name == metric_name
        assert wrapper_config.config.backend.name == "OTHER"

    @pytest.mark.parametrize("metric_name", list(ImplementedCustomMetrics))
    @pytest.mark.parametrize("test_case", TEST_VALID_INPUT_TOKEN, ids=lambda tc: tc.test_name)
    def test_calculate(
        self,
        metric_name: ImplementedCustomMetrics,
        test_case: CalculateGenericTestCase,
    ) -> None:
        wrapper = CustomTokenMetricsWrapper(metric_name=metric_name)

        if not (isinstance(test_case.reference, list) and isinstance(test_case.candidate, list)):
            pytest.skip("CustomTokenMetricsWrapper only works with list inputs")

        result = wrapper.calculate(reference=test_case.reference, candidate=test_case.candidate)
        assert isinstance(result, FScoreMetricResult)
        assert result.metric_name == metric_name
        assert result.reference_length == len(test_case.reference)
        assert result.candidate_length == len(test_case.candidate)
        assert isinstance(result.true_positive, int)
        assert isinstance(result.false_positive, int)
        assert isinstance(result.false_negative, int)
        assert isinstance(result.precision, float)
        assert isinstance(result.recall, float)
        assert isinstance(result.f1, float)
        assert isinstance(result.f2, float)
        assert isinstance(result.f05, float)

    @pytest.mark.parametrize("metric_name", list(ImplementedCustomMetrics))
    @pytest.mark.parametrize("test_case", TEST_VALID_INPUT_SAME_TOKEN, ids=lambda tc: tc.test_name)
    def test_calculate_same_tokens(
        self,
        metric_name: ImplementedCustomMetrics,
        test_case: CalculateGenericTestCase,
    ) -> None:
        wrapper = CustomTokenMetricsWrapper(metric_name=metric_name)

        if not (isinstance(test_case.reference, list) and isinstance(test_case.candidate, list)):
            pytest.skip("CustomTokenMetricsWrapper only works with list inputs")

        result = wrapper.calculate(reference=test_case.reference, candidate=test_case.candidate)
        assert isinstance(result, FScoreMetricResult)
        assert result.metric_name == metric_name
        assert result.reference_length == len(test_case.reference)
        assert result.candidate_length == len(test_case.candidate)

        assert result.true_positive == len(test_case.reference)
        assert result.false_positive == 0
        assert result.false_negative == 0
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0
        assert result.f2 == 1.0
        assert result.f05 == 1.0

    @pytest.mark.parametrize("metric_name", list(ImplementedCustomMetrics))
    @pytest.mark.parametrize("test_case", TEST_INVALID_INPUT_CUSTOM_TOKEN, ids=lambda tc: tc.test_name)
    def test_calculate_invalid_input_custom_token(
        self,
        metric_name: ImplementedCustomMetrics,
        test_case: CustomTokenInvalidInputTestCase,
    ) -> None:
        wrapper = CustomTokenMetricsWrapper(metric_name=metric_name)

        with pytest.raises(test_case.expected_exception):
            wrapper.calculate(reference=test_case.reference, candidate=test_case.candidate)  # type: ignore

    @pytest.mark.parametrize("metric_name", list(ImplementedCustomMetrics))
    @pytest.mark.parametrize(
        "test_case",
        TEST_VALID_INPUT_TOKEN + TEST_VALID_INPUT_SAME_TOKEN,
        ids=lambda tc: tc.test_name,
    )
    def test_calculate_bidirectional(
        self,
        metric_name: ImplementedCustomMetrics,
        test_case: CalculateGenericTestCase,
    ) -> None:
        wrapper = CustomTokenMetricsWrapper(metric_name=metric_name)

        if not (isinstance(test_case.reference, list) and isinstance(test_case.candidate, list)):
            pytest.skip("CustomTokenMetricsWrapper only works with list inputs")

        result1 = wrapper.calculate(reference=test_case.reference, candidate=test_case.candidate)
        result2 = wrapper.calculate(reference=test_case.candidate, candidate=test_case.reference)

        assert isinstance(result1, FScoreMetricResult)
        assert isinstance(result2, FScoreMetricResult)
        assert result1.metric_name == result2.metric_name == metric_name
        assert result1.reference_length == len(test_case.reference)
        assert result1.candidate_length == len(test_case.candidate)
        assert result2.reference_length == len(test_case.candidate)
        assert result2.candidate_length == len(test_case.reference)

        assert 0 <= result1.f1 <= 1.0
        assert 0 <= result2.f1 <= 1.0
        assert 0 <= result1.f2 <= 1.0
        assert 0 <= result2.f2 <= 1.0
        assert 0 <= result1.f05 <= 1.0
        assert 0 <= result2.f05 <= 1.0
