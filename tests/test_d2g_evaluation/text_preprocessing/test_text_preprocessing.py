import pytest

from d2g_evaluation.text_preprocessing.text_preprocessing import TextPreprocessor
from tests.test_d2g_evaluation.text_preprocessing.conftest import (
    TEST_PREPROCESS_METHOD,
    TEST_PREPROCESS_STRING,
    TEST_TOKENIZE_STRING,
    PreprocessStringTestCase,
    PreprocessTestCase,
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

    @pytest.mark.parametrize("test_case", TEST_PREPROCESS_METHOD, ids=lambda tc: tc.test_name)
    def test_preprocess(self, test_case: PreprocessTestCase) -> None:
        preprocessor = TextPreprocessor()
        result = preprocessor.preprocess(
            text=test_case.input_text,
            string_preprocessing_method=test_case.string_preprocessing_method,
            tokenization_method=test_case.tokenization_method,
            n=test_case.n,
        )

        assert result.original == test_case.input_text
        assert result.after_string_preprocessing == test_case.expected_after_string_preprocessing
        assert result.after_tokenization == test_case.expected_after_tokenization
        assert result.string_preprocessing_method == test_case.string_preprocessing_method
        assert result.tokenization_method == test_case.tokenization_method
        assert result.n_param == test_case.expected_n_param
        assert result.is_tokenized == test_case.expected_is_tokenized
        assert result.result == test_case.expected_result
        assert result.config_name == test_case.expected_config_name

    def test_preprocess_no_methods_raises_error(self) -> None:
        """Test that providing no preprocessing methods raises ValueError."""
        preprocessor = TextPreprocessor()

        with pytest.raises(
            ValueError, match="At least one of string_preprocessing_method or tokenization_method must be provided"
        ):
            preprocessor.preprocess(text="Hello")
