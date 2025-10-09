from dataclasses import asdict, dataclass
from enum import StrEnum, unique
from typing import TYPE_CHECKING, Any, ClassVar

import trafilatura

from d2g_extractors.extractors.base_extractor import BaseExtractor, ExtractionResult

if TYPE_CHECKING:
    from collections.abc import Callable


@unique
class TrafilaturaMethods(StrEnum):
    EXTRACT = "extract"
    HTML2TXT = "html2txt"
    BASELINE = "baseline"


@dataclass(slots=True)
class TrafilaturaExtractConfig:
    fast: bool
    favor_precision: bool
    favor_recall: bool
    include_comments: bool
    include_tables: bool
    include_links: bool
    include_images: bool

    @property
    def method_name(self) -> TrafilaturaMethods:
        return TrafilaturaMethods.EXTRACT

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TrafilaturaHtml2TxtConfig:
    clean: bool

    @property
    def method_name(self) -> TrafilaturaMethods:
        return TrafilaturaMethods.HTML2TXT

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TrafilaturaExtractor(BaseExtractor):
    EXTRACTOR_NAME = "trafilatura"

    DEFAULT_EXTRACT_CONFIG_FAST: ClassVar[TrafilaturaExtractConfig] = TrafilaturaExtractConfig(
        fast=True,
        favor_precision=False,
        favor_recall=False,
        include_comments=False,
        include_tables=True,
        include_links=False,
        include_images=False,
    )
    DEFAULT_EXTRACT_CONFIG_NO_FAST: ClassVar[TrafilaturaExtractConfig] = TrafilaturaExtractConfig(
        fast=False,
        favor_precision=False,
        favor_recall=False,
        include_comments=False,
        include_tables=True,
        include_links=False,
        include_images=False,
    )

    DEFAULT_EXTRACT_CONFIG_PRECISION: ClassVar[TrafilaturaExtractConfig] = TrafilaturaExtractConfig(
        fast=False,
        favor_precision=True,
        favor_recall=False,
        include_comments=False,
        include_tables=True,
        include_links=False,
        include_images=False,
    )
    DEFAULT_EXTRACT_CONFIG_RECALL: ClassVar[TrafilaturaExtractConfig] = TrafilaturaExtractConfig(
        fast=False,
        favor_precision=False,
        favor_recall=True,
        include_comments=False,
        include_tables=True,
        include_links=False,
        include_images=False,
    )

    DEFAULT_EXTRACT_CONFIG_COMMENTS: ClassVar[TrafilaturaExtractConfig] = TrafilaturaExtractConfig(
        fast=False,
        favor_precision=False,
        favor_recall=False,
        include_comments=True,
        include_tables=True,
        include_links=False,
        include_images=False,
    )

    DEFAULT_EXTRACT_CONFIG_PRECISION_COMMENTS: ClassVar[TrafilaturaExtractConfig] = TrafilaturaExtractConfig(
        fast=False,
        favor_precision=True,
        favor_recall=False,
        include_comments=True,
        include_tables=True,
        include_links=False,
        include_images=False,
    )

    DEFAULT_EXTRACT_CONFIG_RECALL_COMMENTS: ClassVar[TrafilaturaExtractConfig] = TrafilaturaExtractConfig(
        fast=False,
        favor_precision=False,
        favor_recall=True,
        include_comments=True,
        include_tables=True,
        include_links=False,
        include_images=False,
    )

    DEFAULT_EXTRACT_CONFIG_GREEDY: ClassVar[TrafilaturaExtractConfig] = TrafilaturaExtractConfig(
        fast=False,
        favor_precision=False,
        favor_recall=False,
        include_comments=True,
        include_tables=True,
        include_links=False,
        include_images=True,
    )

    DEFAULT_EXTRACT_CONFIG_GREEDY_RECALL: ClassVar[TrafilaturaExtractConfig] = TrafilaturaExtractConfig(
        fast=False,
        favor_precision=False,
        favor_recall=True,
        include_comments=True,
        include_tables=True,
        include_links=False,
        include_images=True,
    )

    DEFAULT_EXTRACT_CONFIG_GREEDY_PRECISION: ClassVar[TrafilaturaExtractConfig] = TrafilaturaExtractConfig(
        fast=False,
        favor_precision=True,
        favor_recall=False,
        include_comments=True,
        include_tables=True,
        include_links=False,
        include_images=True,
    )

    DEFAULT_HTML2TXT_CONFIG: ClassVar[TrafilaturaHtml2TxtConfig] = TrafilaturaHtml2TxtConfig(
        clean=True,
    )

    def __init__(self) -> None:
        super().__init__()

        self._method_map: dict[TrafilaturaMethods, Callable] = {
            TrafilaturaMethods.BASELINE: self._run_baseline,
            TrafilaturaMethods.HTML2TXT: self._run_html2txt,
            TrafilaturaMethods.EXTRACT: self._run_extract,
        }

    def extract_text_from_html(
        self,
        html_string: str,
        method_name: TrafilaturaMethods,
        **kwargs: Any,  # noqa: ANN401
    ) -> ExtractionResult:
        handler = self._method_map.get(method_name)
        if handler is None:
            msg = f"Method '{method_name}' is not supported by {self.extractor_name} extractor."
            self.logger.error(msg)
            raise ValueError(msg)
        return handler(html_string, **kwargs)

    def _run_baseline(self, html_string: str, **kwargs: Any) -> ExtractionResult:  # noqa: ANN401
        _ = kwargs
        _, result, _ = trafilatura.baseline(filecontent=html_string)

        return ExtractionResult(
            extractor_name=self.extractor_name,
            version=self.version,
            extractor_method=TrafilaturaMethods.BASELINE,
            extractor_params={},
            extraction_result=result,
        )

    def _run_html2txt(self, html_string: str, **kwargs: Any) -> ExtractionResult:  # noqa: ANN401
        result = trafilatura.html2txt(html_string, **kwargs)
        return ExtractionResult(
            extractor_name=self.extractor_name,
            version=self.version,
            extractor_method=TrafilaturaMethods.HTML2TXT,
            extractor_params=kwargs,
            extraction_result=result,
        )

    def _run_extract(self, html_string: str, **kwargs: Any) -> ExtractionResult:  # noqa: ANN401
        result = trafilatura.extract(html_string, **kwargs)

        return ExtractionResult(
            extractor_name=self.extractor_name,
            version=self.version,
            extractor_method=TrafilaturaMethods.EXTRACT,
            extractor_params=kwargs,
            extraction_result=result,
        )
