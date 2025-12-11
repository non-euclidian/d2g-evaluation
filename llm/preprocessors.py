import re
import unicodedata

from bs4 import BeautifulSoup
from json_repair import repair_json


class FaultTolerantJsonPreprocessor:
    """
    Preprocesses potentially malformed JSON coming from an LLM
    into a form suitable for the Python parser.

    """

    def process(self, raw_annotations: str) -> str:
        return_value = repair_json(raw_annotations, ensure_ascii=False)
        return_value = unicodedata.normalize("NFKC", return_value)
        # remove control chars that can break potentially break JSON parsing.
        return_value = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\x80-\x9F\u2028\u2029]", "", return_value)
        # zero-width space
        return_value = re.sub(r"[\u00a0\u200b\u200c\u200d\ufeff]", "", return_value)
        return_value = return_value.replace("\xa0", " ")

        # pretty-printing like ```python, ```jsonl, etc.
        def strip_code_fences(text: str) -> str:
            pattern = r"```[a-zA-Z0-9]*\n|\n```$|^```[a-zA-Z0-9]*\r?\n|\r?\n```$"
            return re.sub(pattern, "", text).strip()

        return_value = strip_code_fences(return_value)

        return return_value.strip()


class HtmlRemover:
    """
    Strips HTML tags sometimes returned as part of annotation by an LLM.
    """

    def process(self, text: str) -> str:
        soup = BeautifulSoup(text, "html.parser")

        return soup.get_text(separator=" ", strip=True)


class ReadabilityHtmlDenoiser:
    """
    Extracts main html document content to reduce annotation costs
    and potentially improve LLM annotation quality.
    This implementation is based on [Readability](https://github.com/buriy/python-readability)

    """

    def process(self, html: str) -> str:
        raise NotImplementedError
