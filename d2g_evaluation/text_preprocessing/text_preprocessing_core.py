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
    def result(self) -> str | list[str]:
        """Get the final result - tokens if available, otherwise processed text."""
        return self.after_tokenization if self.after_tokenization is not None else self.after_string_preprocessing

    @property
    def is_tokenized(self) -> bool:
        """Check if the result contains tokens."""
        return self.after_tokenization is not None

    @property
    def config_name(self) -> str:
        """Get a human-readable string representation of the preprocessing configuration."""
        str_prepr = self.string_preprocessing_method.value if self.string_preprocessing_method else "none"
        tok_prepr = self.tokenization_method.value if self.tokenization_method else "none"
        n = self.n_param if self.n_param else "none"
        return f"str_{str_prepr}__tok_{tok_prepr}__n_{n}"
