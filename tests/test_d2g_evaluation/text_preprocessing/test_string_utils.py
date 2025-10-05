import pytest

from d2g_evaluation.text_preprocessing.string_utils import StringUtils
from tests.test_d2g_evaluation.text_preprocessing.conftest import (
    TEST_DECODE_UNICODE_ESCAPES,
    TEST_NORMALIZE_STRING,
    TEST_NORMALIZE_WHITESPACES,
    TEST_TOKENIZE_BY_CHAR_NGRAMS,
    TEST_TOKENIZE_BY_NCHARS,
)


class TestStringUtils:
    @pytest.mark.parametrize(("input_text", "expected"), TEST_DECODE_UNICODE_ESCAPES)
    def test_decode_unicode_escapes(self, input_text: str, expected: str) -> None:
        result = StringUtils.decode_unicode_escapes(text=input_text)
        assert result == expected

    @pytest.mark.parametrize("raw_input", [pytest.param(r"\u276", id="incomplete-unicode")])
    def test_decode_unicode_escapes_handles_malformed_gracefully(self, raw_input: str) -> None:
        result = StringUtils.decode_unicode_escapes(text=raw_input)
        assert result is not None
        assert isinstance(result, str)

    @pytest.mark.parametrize(("raw_input", "expected_output", "replacement", "strip"), TEST_NORMALIZE_WHITESPACES)
    def test_normalize_whitespaces(
        self, *, raw_input: str, expected_output: str, replacement: str, strip: bool
    ) -> None:
        result = StringUtils.normalize_whitespaces(text=raw_input, replacement=replacement, strip=strip)
        assert result == expected_output, f"Expected {expected_output}, but got {result} for input {raw_input}"

    @pytest.mark.parametrize(("raw_input", "expected_output", "replacement", "strip"), TEST_NORMALIZE_STRING)
    def test_normalized_string(self, *, raw_input: str, expected_output: str, replacement: str, strip: bool) -> None:
        result = StringUtils.normalize_string(text=raw_input, replacement=replacement, strip=strip)
        assert result == expected_output, f"Expected {expected_output}, but got {result} for input {raw_input}"

    @pytest.mark.parametrize(("raw_input", "n", "expected_output"), TEST_TOKENIZE_BY_CHAR_NGRAMS)
    def test_tokenize_by_char_ngrams(self, *, raw_input: str, n: int, expected_output: list[str]) -> None:
        result = StringUtils.tokenize_by_char_ngrams(text=raw_input, n=n)
        assert result == expected_output, f"Expected {expected_output}, but got {result} for input {raw_input}"

    def test_tokenize_by_char_ngrams_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="Expected str, but got"):
            StringUtils.tokenize_by_char_ngrams(text=123, n=3)  # type: ignore

        with pytest.raises(TypeError, match="Expected str, but got"):
            StringUtils.tokenize_by_char_ngrams(text=123.45, n=3)  # type: ignore

        with pytest.raises(TypeError, match="Expected str, but got"):
            StringUtils.tokenize_by_char_ngrams(text=["hello"], n=3)  # type: ignore

        with pytest.raises(TypeError, match="Expected str, but got"):
            StringUtils.tokenize_by_char_ngrams(text={"hello"}, n=3)  # type: ignore

        with pytest.raises(ValueError, match="n must be greater than 0, but got"):
            StringUtils.tokenize_by_char_ngrams(text="hello", n=0)

        with pytest.raises(ValueError, match="n must be greater than 0, but got"):
            StringUtils.tokenize_by_char_ngrams(text="hello", n=-1)

    @pytest.mark.parametrize(("raw_input", "n", "expected_output"), TEST_TOKENIZE_BY_NCHARS)
    def test_tokenize_by_nchars(self, *, raw_input: str, n: int, expected_output: list[str]):
        result = StringUtils.tokenize_by_nchars(text=raw_input, n=n)
        assert result == expected_output, f"Expected {expected_output}, but got {result} for input {raw_input}"

    def test_tokenize_by_nchars_invalid_type(self):
        with pytest.raises(TypeError, match="Expected str, but got"):
            StringUtils.tokenize_by_nchars(text=123, n=3)

        with pytest.raises(TypeError, match="Expected str, but got"):
            StringUtils.tokenize_by_nchars(text=123.45, n=3)

        with pytest.raises(TypeError, match="Expected str, but got"):
            StringUtils.tokenize_by_nchars(text=["hello"], n=3)

        with pytest.raises(TypeError, match="Expected str, but got"):
            StringUtils.tokenize_by_nchars(text={"hello"}, n=3)

        with pytest.raises(ValueError, match="n must be greater than 0, but got"):
            StringUtils.tokenize_by_nchars(text="hello", n=0)

        with pytest.raises(ValueError, match="n must be greater than 0, but got"):
            StringUtils.tokenize_by_nchars(text="hello", n=-1)
