import re

PATTERN_ALL_WHITESPACE_CHARS = re.compile(r"\s+")


class StringUtils:
    @staticmethod
    def decode_unicode_escapes(text: str) -> str:
        try:
            # First attempt: strict decoding (will fail on malformed escapes)
            return text.encode(encoding="raw_unicode_escape", errors="strict").decode(
                encoding="unicode_escape", errors="strict"
            )
        except (UnicodeDecodeError, UnicodeEncodeError, AttributeError):
            # Fallback: relaxed decoding
            print("Warning: Failed to decode unicode escapes in text, trying relaxed mode.")
            print(f"Text: {text!r}")
            print("Attempting to decode with 'backslashreplace' and 'ignore' error handling.")
            try:
                return text.encode(encoding="raw_unicode_escape", errors="backslashreplace").decode(
                    encoding="unicode_escape", errors="ignore"
                )
            except Exception as relaxed_failure:
                msg = f"Failed to decode/encode unicode escapes in text even with relaxed mode: {text!r}"
                raise ValueError(msg) from relaxed_failure

    @staticmethod
    def normalize_whitespaces(*, text: str, replacement: str = " ", strip: bool = True) -> str:
        text = re.sub(PATTERN_ALL_WHITESPACE_CHARS, replacement, text)
        if strip:
            text = text.strip()
        return text

    @staticmethod
    def normalize_string(*, text: str, replacement: str = " ", strip: bool = True) -> str:
        text = StringUtils.decode_unicode_escapes(text=text)
        text = StringUtils.normalize_whitespaces(text=text, replacement=replacement, strip=strip)
        return text  # noqa: RET504

    @staticmethod
    def _validate_tokenizer_input(text: str, n: int) -> None:
        """Private helper to validate input for tokenizer functions."""
        if not isinstance(text, str):
            msg = f"Expected str, but got {type(text)}"
            raise TypeError(msg)
        if n < 1:
            msg = f"n must be greater than 0, but got {n}"
            raise ValueError(msg)

    @staticmethod
    def tokenize_by_char_ngrams(text: str, n: int = 3) -> list[str]:
        """
        Generates overlapping character n-grams from a string (sliding window).
        Example: ('hello', 3) -> ['hel', 'ell', 'llo']
        """
        StringUtils._validate_tokenizer_input(text, n)

        if len(text) < n:
            return [text] if text else []
        return [text[i : i + n] for i in range(len(text) - n + 1)]

    @staticmethod
    def tokenize_by_nchars(text: str, n: int = 3) -> list[str]:
        """
        Splits a string into non-overlapping chunks of size n.
        Example: ('hello world', 3) -> ['hel', 'lo ', 'wor', 'ld']
        """
        StringUtils._validate_tokenizer_input(text, n)
        if not text:
            return []
        return [text[i : i + n] for i in range(0, len(text), n)]
