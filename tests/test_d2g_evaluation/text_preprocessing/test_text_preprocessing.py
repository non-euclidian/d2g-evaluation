import pytest

from d2g_evaluation.text_preprocessing.text_preprocessing import TextPreprocessor
from d2g_evaluation.text_preprocessing.text_preprocessing_core import (
    ImplementedStringPreprocessing,
    ImplementedTokenization,
)
from tests.test_d2g_evaluation.text_preprocessing.conftest import TEST_PREPROCESS_STRING, TEST_TOKENIZE_STRING


class TestTextPreprocessor:
    @pytest.mark.parametrize(("raw_input", "expected_output", "string_preprocessing_method"), TEST_PREPROCESS_STRING)
    def test_preprocess_string(
        self, *, raw_input: str, expected_output: str, string_preprocessing_method: ImplementedStringPreprocessing
    ):
        result = TextPreprocessor()._preprocess_string(
            text=raw_input, string_preprocessing_method=string_preprocessing_method
        )
        assert result == expected_output, f"Expected {expected_output}, but got {result} for input {raw_input}"

    @pytest.mark.parametrize(("raw_input", "tokenization_method", "n", "expected_output"), TEST_TOKENIZE_STRING)
    def test_tokenize_string(
        self, *, raw_input: str, tokenization_method: ImplementedTokenization, n: int, expected_output: list[str]
    ):
        result = TextPreprocessor()._tokenize_string(text=raw_input, tokenization_method=tokenization_method, n=n)
        assert result == expected_output, f"Expected {expected_output}, but got {result} for input {raw_input}"
