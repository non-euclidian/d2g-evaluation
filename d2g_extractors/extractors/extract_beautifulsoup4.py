from enum import StrEnum, unique

import bs4  # noqa: F401

from d2g_extractors.extractors.base_extractor import BaseExtractor


@unique
class BeautifulSoup4Methods(StrEnum):
    GET_TEXT = "get_text"


class BeautifulSoup4Extractor(BaseExtractor):
    EXTRACTOR_NAME = "beautifulsoup4"
