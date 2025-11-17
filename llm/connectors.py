import json
import unicodedata
from abc import ABC, abstractmethod
from typing import Any

import aiohttp


class RetryableAPIError(Exception):
    """Signals that request may succeed if retried."""


class LLMConnector(ABC):
    """Abstract connector base class"""

    def __init__(self, base_url: str, api_key: str, config: dict[str, Any]) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.config = config

    @abstractmethod
    def _create_payload_from(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Converts the doc into the JSON payload understood by remote API."""
        ...

    @abstractmethod
    def _parse_response(self, response_text: str) -> dict[str, Any]:
        """Extracts annotations from API response."""
        ...

    async def call_llm(
        self,
        session: aiohttp.ClientSession,
        doc: dict[str, Any],
    ) -> dict[str, Any]:
        payload = self._create_payload_from(doc)

        async with session.post(
            self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=120,  # type: ignore[arg-type]
        ) as r:
            response_text = await r.text()
            if r.status == 200:  # noqa PLR2004 No magic: HTTP status code 200 is a well-known number.
                return self._parse_response(response_text)

            if r.status in (429, 500, 502, 503, 504):
                raise RetryableAPIError(f"Retryable API error {r.status}: {response_text}")  # noqa
            raise Exception(f"Non-retryable API error {r.status}: {response_text}")  # noqa

    def _preprocess_annotations(self, raw_annotations: str) -> str:
        "Remove LLM formatting that breaks JSON parsing if present"

        return_value = unicodedata.normalize("NFKC", raw_annotations)
        # Markdown formatting, zero-width space
        replacements = ["```json", "```", "\u200b"]
        for r in replacements:
            return_value = return_value.replace(r, "")

        return return_value.strip()


class OpenRouterConnector(LLMConnector):
    """Gathers annotations from OpenRouter.com"""

    def __init__(self, api_key: str, config: dict[str, Any]) -> None:
        super().__init__("https://openrouter.ai/api/v1/chat/completions", api_key, config)

    def _create_payload_from(self, doc: dict[str, Any]) -> dict[str, Any]:
        return {
            "model": self.config["model"],
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": self.config["prompt"]["system"]},
                {"role": "user", "content": self.config["prompt"]["user"].format(html=doc["html"])},
            ],
            "provider": {
                # Gotcha: Reproducibility. Good for debugging,
                # but different providers may be using different quantizations,
                # leading to different results across runs.
                "sort": "latency",
            },
        }

    def _parse_response(self, response_text: str) -> dict[str, Any]:
        try:
            data = json.loads(response_text)
            model_response = data["choices"][0]["message"]["content"]
            annotations = self._preprocess_annotations(model_response)
            annotations = json.loads(annotations)["annotations"]
            total_tokens = data["usage"]["prompt_tokens"]

        except Exception as e:
            raise ValueError(f"Couldn't process API server response:\n{response_text}") from e  # noqa

        return {
            "annotations": annotations,
            "total_tokens": total_tokens,
        }
