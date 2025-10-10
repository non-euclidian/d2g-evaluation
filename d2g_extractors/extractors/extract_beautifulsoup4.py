import inspect
from dataclasses import asdict, dataclass
from enum import StrEnum, unique
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar

import bs4

from d2g_extractors.extractors.base_extractor import BaseExtractor, ExtractionResult

if TYPE_CHECKING:
    from collections.abc import Callable


@unique
class BeautifulSoup4Methods(StrEnum):
    GET_TEXT = "get_text"


@dataclass(slots=True)
class BeautifulSoup4GetTextConfig:
    features: str
    separator: str
    strip: bool

    @property
    def method_name(self) -> BeautifulSoup4Methods:
        return BeautifulSoup4Methods.GET_TEXT

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BeautifulSoup4Extractor(BaseExtractor):
    EXTRACTOR_NAME = "beautifulsoup4"

    DEFAULT_GET_TEXT_CONFIG: ClassVar[BeautifulSoup4GetTextConfig] = BeautifulSoup4GetTextConfig(
        features="html.parser",
        separator=" ",
        strip=False,
    )

    DEFAULT_GET_TEXT_CONFIG_LXML: ClassVar[BeautifulSoup4GetTextConfig] = BeautifulSoup4GetTextConfig(
        features="lxml",
        separator=" ",
        strip=False,
    )

    def __init__(self) -> None:
        super().__init__()

        self._method_map: dict[BeautifulSoup4Methods, Callable] = {
            BeautifulSoup4Methods.GET_TEXT: self._run_get_text,
        }

    @cached_property
    def _bs_param_names(self) -> set[str]:
        return set(inspect.signature(bs4.BeautifulSoup).parameters)

    @cached_property
    def _get_text_param_names(self) -> set[str]:
        return set(inspect.signature(bs4.element.Tag.get_text).parameters)

    def extract_text_from_html(
        self,
        html_string: str,
        method_name: BeautifulSoup4Methods,
        **kwargs: Any,  # noqa: ANN401
    ) -> ExtractionResult:
        handler = self._method_map.get(method_name)
        if handler is None:
            msg = f"Method '{method_name}' is not supported by {self.extractor_name} extractor."
            self.logger.error(msg)
            raise ValueError(msg)

        return handler(html_string, **kwargs)

    def _run_get_text(self, html_string: str, **kwargs: Any) -> ExtractionResult:  # noqa: ANN401
        bs_kwargs, get_text_kwargs = self._split_kwargs(**kwargs)

        soup = bs4.BeautifulSoup(html_string, **bs_kwargs)
        result = soup.get_text(**get_text_kwargs)

        return ExtractionResult(
            extractor_name=self.extractor_name,
            version=self.version,
            extractor_method=BeautifulSoup4Methods.GET_TEXT,
            extractor_params={"bs_kwargs": bs_kwargs, "get_text_kwargs": get_text_kwargs},
            extraction_result=result,
        )

    def _split_kwargs(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:  # noqa: ANN401
        """Split kwargs into those for BeautifulSoup() and for get_text()."""
        bs_kwargs = {k: v for k, v in kwargs.items() if k in self._bs_param_names}
        get_text_kwargs = {k: v for k, v in kwargs.items() if k in self._get_text_param_names}

        # warn about unknown args
        unknown = set(kwargs) - self._bs_param_names - self._get_text_param_names
        if unknown:
            msg = f"Unknown kwargs: {unknown}"
            self.logger.warning(msg)

        return bs_kwargs, get_text_kwargs
