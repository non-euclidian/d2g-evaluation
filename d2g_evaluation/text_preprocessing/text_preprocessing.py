import logging

from d2g_evaluation.text_preprocessing.string_utils import StringUtils
from d2g_evaluation.text_preprocessing.text_preprocessing_core import (
    ImplementedStringPreprocessing,
    ImplementedTokenization,
    TextPreprocessingResult,
)


class TextPreprocessor:
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def _preprocess_string(
        self,
        text: str,
        string_preprocessing_method: ImplementedStringPreprocessing,
    ) -> str:
        match string_preprocessing_method:
            case ImplementedStringPreprocessing.DECODE_UNICODE_ESCAPES:
                return StringUtils.decode_unicode_escapes(text=text)
            case ImplementedStringPreprocessing.NORMALIZE_WHITESPACES:
                return StringUtils.normalize_whitespaces(text=text)
            case ImplementedStringPreprocessing.NORMALIZE_STRING:
                return StringUtils.normalize_string(text=text, replacement=" ", strip=True)
            case ImplementedStringPreprocessing.REMOVE_WHITESPACES:
                result_str = StringUtils.decode_unicode_escapes(text=text)
                return StringUtils.normalize_whitespaces(text=result_str, replacement="", strip=True)
            case _:
                msg = f"Unsupported string preprocessing method: {string_preprocessing_method}"
                self.logger.error(msg)
                raise ValueError(msg)

    def _tokenize_string(
        self,
        text: str,
        tokenization_method: ImplementedTokenization,
        n: int = 3,
    ) -> list[str]:
        match tokenization_method:
            case ImplementedTokenization.CHAR_NGRAMS:
                return StringUtils.tokenize_by_char_ngrams(text=text, n=n)
            case ImplementedTokenization.NCHARS:
                return StringUtils.tokenize_by_nchars(text=text, n=n)
            case _:
                msg = f"Unsupported tokenization method: {tokenization_method}"
                self.logger.error(msg)
                raise ValueError(msg)

    def preprocess(
        self,
        text: str,
        string_preprocessing_method: ImplementedStringPreprocessing | None = None,
        tokenization_method: ImplementedTokenization | None = None,
        n: int = 3,
    ) -> TextPreprocessingResult:
        if not isinstance(text, str):
            msg = f"Input text must be a string, got {type(text)}"
            self.logger.error(msg)
            raise TypeError(msg)

        if string_preprocessing_method is None:
            processed_text = text
        else:
            processed_text = self._preprocess_string(text=text, string_preprocessing_method=string_preprocessing_method)

        tokens = None
        if tokenization_method is not None:
            tokens = self._tokenize_string(text=processed_text, tokenization_method=tokenization_method, n=n)

        return TextPreprocessingResult(
            original=text,
            after_string_preprocessing=processed_text,
            after_tokenization=tokens,
            string_preprocessing_method=string_preprocessing_method,
            tokenization_method=tokenization_method,
            n_param=n if tokenization_method is not None else None,
        )
