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

    original: str | list[str]
    after_string_preprocessing: str | None = None
    after_tokenization: list[str] | None = None
    string_preprocessing_method: ImplementedStringPreprocessing | str | None = None
    tokenization_method: ImplementedTokenization | str | None = None
    n_param: int | None = None

    @property
    def result(self) -> str | list[str]:
        """Get the final result - tokens if available, otherwise processed text, otherwise original."""
        if self.after_tokenization is not None:
            return self.after_tokenization
        if self.after_string_preprocessing is not None:
            return self.after_string_preprocessing
        return self.original

    @property
    def is_tokenized(self) -> bool:
        """Check if the result contains tokens."""
        return self.after_tokenization is not None

    @property
    def is_list_input(self) -> bool:
        """Check if the original input was a list."""
        return isinstance(self.original, list)

    @property
    def config_name(self) -> str:
        """Get a human-readable string representation of the preprocessing configuration."""
        # handle string preprocessing method
        if self.string_preprocessing_method is not None:
            if isinstance(self.string_preprocessing_method, ImplementedStringPreprocessing):
                str_prepr = self.string_preprocessing_method.value
            else:
                str_prepr = str(self.string_preprocessing_method)
        else:
            str_prepr = None

        # Handle tokenization method
        if self.tokenization_method is not None:
            if isinstance(self.tokenization_method, ImplementedTokenization):
                tok_prepr = self.tokenization_method.value
            else:
                tok_prepr = str(self.tokenization_method)
        else:
            tok_prepr = None

        n = self.n_param if self.n_param else None

        # Add input type indicator
        input_type = "list" if self.is_list_input else "str"

        return f"input_{input_type}__str_{str_prepr}__tok_{tok_prepr}__n_{n}"
