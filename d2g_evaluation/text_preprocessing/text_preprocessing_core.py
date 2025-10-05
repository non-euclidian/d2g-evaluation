from dataclasses import dataclass
from enum import StrEnum, unique


@unique
class ImplementedStringPreprocessing(StrEnum):
    DECODE_UNICODE_ESCAPES = "decode_unicode_escapes"
    NORMALIZE_WHITESPACES = "normalize_whitespaces"
    NORMALIZE_STRING = "normalize_string"
    REMOVE_WHITESPACES = "remove_whitespaces"


@unique
class ImplementedTokenization(StrEnum):
    CHAR_NGRAMS = "char_ngrams"
    NCHARS = "nchars"


@dataclass(slots=True)
class TextPreprocessingResult:
    """Result of text preprocessing operation."""

    original: str
    after_string_preprocessing: str
    after_tokenization: list[str] | None = None
    string_preprocessing_method: ImplementedStringPreprocessing | None = None
    tokenization_method: ImplementedTokenization | None = None
    n_param: int | None = None

    @property
    def is_tokenized(self) -> bool:
        """Check if the result contains tokens."""
        return self.after_tokenization is not None

    @property
    def result(self) -> str | list[str]:
        """Get the final result - tokens if available, otherwise processed text."""
        return self.after_tokenization if self.after_tokenization is not None else self.after_string_preprocessing
