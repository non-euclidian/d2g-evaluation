import logging
from abc import ABC
from importlib.metadata import version
from typing import ClassVar


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
