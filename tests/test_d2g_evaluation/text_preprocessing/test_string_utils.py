import pytest

from d2g_evaluation.text_preprocessing.string_utils import StringUtils
from tests.test_d2g_evaluation.text_preprocessing.conftest import (
    TEST_DECODE_UNICODE_ESCAPES,
    TEST_NORMALIZE_STRING,
    TEST_NORMALIZE_WHITESPACES,
    TEST_TOKENIZE_BY_CHAR_NGRAMS,
    TEST_TOKENIZE_BY_NCHARS,
    DecodeUnicodeTestCase,
    NormalizeStringTestCase,
    NormalizeWhitespacesTestCase,
    TokenizeByCharNgramsTestCase,
    TokenizeByNcharsTestCase,
)


class TestStringUtils:
    @pytest.mark.parametrize("test_case", TEST_DECODE_UNICODE_ESCAPES, ids=lambda tc: tc.test_name)
    def test_decode_unicode_escapes(self, test_case: DecodeUnicodeTestCase) -> None:
        result = StringUtils.decode_unicode_escapes(text=test_case.input_text)
        assert result == test_case.expected_text, (
            f"Expected {test_case.expected_text!r}, but got {result!r} for input {test_case.input_text!r}"
        )

    @pytest.mark.parametrize("raw_input", [pytest.param(r"\u276", id="incomplete-unicode")])
    def test_decode_unicode_escapes_handles_malformed_gracefully(self, raw_input: str) -> None:
        result = StringUtils.decode_unicode_escapes(text=raw_input)
        assert result is not None
        assert isinstance(result, str)

    @pytest.mark.parametrize("test_case", TEST_NORMALIZE_WHITESPACES, ids=lambda tc: tc.test_name)
    def test_normalize_whitespaces(self, test_case: NormalizeWhitespacesTestCase) -> None:
        result = StringUtils.normalize_whitespaces(
            text=test_case.input_text, replacement=test_case.replacement, strip=test_case.strip
        )
        assert result == test_case.expected_text, (
            f"Expected {test_case.expected_text!r}, but got {result!r} for input {test_case.input_text!r}"
        )

    @pytest.mark.parametrize("test_case", TEST_NORMALIZE_STRING, ids=lambda tc: tc.test_name)
    def test_normalize_string(self, test_case: NormalizeStringTestCase) -> None:
        result = StringUtils.normalize_string(
            text=test_case.input_text, replacement=test_case.replacement, strip=test_case.strip
        )
        assert result == test_case.expected_text, (
            f"Expected {test_case.expected_text!r}, but got {result!r} for input {test_case.input_text!r}"
        )

    @pytest.mark.parametrize("test_case", TEST_TOKENIZE_BY_CHAR_NGRAMS, ids=lambda tc: tc.test_name)
    def test_tokenize_by_char_ngrams(self, test_case: TokenizeByCharNgramsTestCase) -> None:
        result = StringUtils.tokenize_by_char_ngrams(text=test_case.input_text, n=test_case.n)
        assert result == test_case.expected_tokens, (
            f"Expected {test_case.expected_tokens!r}, but got {result!r} for input {test_case.input_text!r} with n={test_case.n}"
        )

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

    @pytest.mark.parametrize("test_case", TEST_TOKENIZE_BY_NCHARS, ids=lambda tc: tc.test_name)
    def test_tokenize_by_nchars(self, test_case: TokenizeByNcharsTestCase) -> None:
        result = StringUtils.tokenize_by_nchars(text=test_case.input_text, n=test_case.n)
        assert result == test_case.expected_tokens, (
            f"Expected {test_case.expected_tokens!r}, but got {result!r} for input {test_case.input_text!r} with n={test_case.n}"
        )

    def test_tokenize_by_nchars_invalid_type(self) -> None:
        with pytest.raises(TypeError, match="Expected str, but got"):
            StringUtils.tokenize_by_nchars(text=123, n=3)  # type: ignore

        with pytest.raises(TypeError, match="Expected str, but got"):
            StringUtils.tokenize_by_nchars(text=123.45, n=3)  # type: ignore

        with pytest.raises(TypeError, match="Expected str, but got"):
            StringUtils.tokenize_by_nchars(text=["hello"], n=3)  # type: ignore

        with pytest.raises(TypeError, match="Expected str, but got"):
            StringUtils.tokenize_by_nchars(text={"hello"}, n=3)  # type: ignore

        with pytest.raises(ValueError, match="n must be greater than 0, but got"):
            StringUtils.tokenize_by_nchars(text="hello", n=0)

        with pytest.raises(ValueError, match="n must be greater than 0, but got"):
            StringUtils.tokenize_by_nchars(text="hello", n=-1)
