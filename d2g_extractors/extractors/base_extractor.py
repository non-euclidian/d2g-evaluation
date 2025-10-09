import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib.metadata import version
from typing import Any, ClassVar


@dataclass(slots=True)
class ExtractionResult:
    extractor_name: str
    version: str

    extraction_result: Any  # checked by __post_init__

    extractor_method: str
    extractor_params: dict[str, Any] = field(default_factory=dict)

    is_successful: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.extraction_result, str):
            self.is_successful = True
        else:
            self.is_successful = False
            self.extraction_result = ""


class BaseExtractor(ABC):
    EXTRACTOR_NAME: ClassVar[str]

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    @property
    def extractor_name(self) -> str:
        """Return the name of the extractor."""
        return self.EXTRACTOR_NAME

    @property
    def version(self) -> str:
        """Return the version of the extractor."""
        return version(self.EXTRACTOR_NAME)

    @abstractmethod
    def extract_text_from_html(self, html_string: str, method_name: Any, **kwargs: Any) -> ExtractionResult: ...  # noqa: ANN401
