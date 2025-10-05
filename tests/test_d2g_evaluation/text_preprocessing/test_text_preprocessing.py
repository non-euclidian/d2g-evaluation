import pytest

from d2g_evaluation.text_preprocessing.text_preprocessing import TextPreprocessor
from tests.test_d2g_evaluation.text_preprocessing.conftest import (
    TEST_PREPROCESS_STRING,
    TEST_TOKENIZE_STRING,
    PreprocessStringTestCase,
    TokenizeStringTestCase,
)


class TestTextPreprocessor:
    @pytest.mark.parametrize("test_case", TEST_PREPROCESS_STRING, ids=lambda tc: tc.test_name)
    def test_preprocess_string(self, test_case: PreprocessStringTestCase) -> None:
        result = TextPreprocessor()._preprocess_string(
            text=test_case.input_text, string_preprocessing_method=test_case.preprocessing_method
        )
        assert result == test_case.expected_text, (
            f"Expected {test_case.expected_text!r}, but got {result!r} for input {test_case.input_text!r}"
        )

    @pytest.mark.parametrize("test_case", TEST_TOKENIZE_STRING, ids=lambda tc: tc.test_name)
    def test_tokenize_string(self, test_case: TokenizeStringTestCase) -> None:
        result = TextPreprocessor()._tokenize_string(
            text=test_case.input_text,
            tokenization_method=test_case.tokenization_method,
            n=test_case.n,
        )
        assert result == test_case.expected_tokens, (
            f"Expected {test_case.expected_tokens!r}, but got {result!r} for input {test_case.input_text!r}"
        )
