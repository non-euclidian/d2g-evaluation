"""
Integrates with an external LLM API in a resilient async manner
to annotate main content text in a given HTML file collection.

Inputs:
- A `jsonl` or `parquet` file with HTML docs;
    Each entry is expected to have two columns: 'task_id' (the unique html doc id in our notation) and 'html'.
- A config file specifying annotation run settings for experiment reproducibility.
    Example .config is included in containing directory.

Outputs:
- `results.jsonl`: LLM-generated main page content annotations;
- `failures.jsonl`: Failed document log for debugging.

Notes:
- Outputs are stored in llm/.annotations/{config_name}/{output}.jsonl;
- Requests are asynchronous to reduce time spent waiting for results;
- This annotator has built-in automatic retry logic with exponential backoff and max retry limit;
- Max concurrent requests are capped (configurable) to avoid API throttling;
- If restarted after a crash, it will process only yet unannotated HTML docs from the input file to save API credits and wall time.\
    This is achieved by checking the documents already present in `results.jsonl` via the `task_id`.\
    Deleting this file will cause the entire HTML collection to be reprocessed from scratch.

"""

import argparse
import asyncio
import json
import logging
import os
import time
import traceback
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiohttp
import polars as pl
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

# ============ Configuration ============
API_KEY = os.getenv("LLM_API_KEY")
ANNOTATION_FILE = "results.jsonl"
FAILURES_FILE = "failures.jsonl"
CONCURRENT_REQUESTS = 5
MAX_RETRIES = 5

# ======================================


# ---------- Logging ----------
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ---------- Helpers ----------


def load_config(config_path: str) -> dict[str, Any]:
    """
    Load reproducible run configuration from a JSON file.

    Expected keys:
      - model (str)
      - connector (str), e.g. 'openrouter'
      - prompt (dict), including 'system' and 'user' prompts

    Optional keys:
      - concurrent_requests (int). Used to prevent rate limiting. The default value is set in CONCURRENT_REQUESTS

    Args:
        config_path (str): Path to the JSON config file.

    Raises:
        ValueError: If required keys are missing or have wrong types.
    """
    p = Path(config_path)
    with p.open(encoding="utf-8") as f:
        config = json.load(f)

    schema = {
        "model": str,
        "connector": str,
        "concurrent_requests": int,
        "prompt": dict,
    }

    if "concurrent_requests" not in config:
        config["concurrent_requests"] = CONCURRENT_REQUESTS

    for key, expected_type in schema.items():
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")  # noqa
        # Convert to the right data type in-place
        try:
            config[key] = expected_type(config[key])
        except (ValueError, TypeError) as err:
            raise ValueError(  # noqa
                f"Config key {key} must be of type {expected_type.__name__}, got {config[key]}"  # noqa
            ) from err

    config["name"] = p.name.replace(".config", "")

    return config


def get_output_path_for_config(filename: str, config: dict[str, Any]) -> str:
    return_value = Path("llm/.annotations") / config["name"] / filename
    # create missing directories if they don't exist yet.
    # if they exist, do nothing.
    return_value.parent.mkdir(parents=True, exist_ok=True)

    return return_value


def load_docs(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if p.suffix == ".parquet":
        df = pl.read_parquet(path)
        return df.to_dicts()
    # Assume JSONL
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_processed_ids(path: str) -> set[str]:
    """Return task_ids already processed successfully, so that they can be skipped."""
    p = Path(path)
    if not p.exists():
        return set()
    with p.open(encoding="utf-8") as f:
        processed = set()
        for line in f:
            try:
                processed.add(json.loads(line)["task_id"])
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed JSON line in {path}")  # noqa G004
        return processed


def save_jsonl_line(obj: dict, path: str) -> None:
    """Append a single JSON object as a line."""
    p = Path(path)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def preprocess_annotations(raw_annotations: str) -> str:
    "Remove LLM formatting that breaks JSON parsing if present"

    return_value = unicodedata.normalize("NFKC", raw_annotations)
    # Markdown formatting, zero-width space
    replacements = ["```json", "```", "\u200b"]
    for r in replacements:
        return_value = return_value.replace(r, "")

    return return_value.strip()


def extract_annotations(response_text: str) -> dict[str, Any]:
    return_value = ""
    try:
        data = json.loads(response_text)
        model_response = data["choices"][0]["message"]["content"]
        return_value = preprocess_annotations(model_response)
        return_value = json.loads(return_value)["annotations"]

    except Exception as e:
        raise ValueError(f"Couldn't process API server response:\n{response_text}") from e  # noqa

    return return_value


# ---------- API Call ----------


class RetryableAPIError(Exception):
    pass


@retry(
    wait=wait_exponential(multiplier=3, min=5, max=30),
    stop=stop_after_attempt(MAX_RETRIES),
    retry=retry_if_exception_type(RetryableAPIError),
)
async def call_llm(session: aiohttp.ClientSession, doc: dict[str, Any], config: dict[str, Any]) -> str:
    payload = {
        "model": config["model"],
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": config["prompt"]["system"]},
            {"role": "user", "content": config["prompt"]["user"].format(html=doc["html"])},
        ],
        "provider": {
            # Gotcha: Reproducibility. Good for debugging,
            # but different providers may be using different quantizations,
            # leading to different results across runs.
            "sort": "latency",
        },
    }

    async with session.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=120,  # type: ignore[arg-type]
    ) as r:
        response_text = await r.text()
        if r.status == 200:  # noqa PLR2004 No magic: HTTP status code 200 is a well-known number.
            return response_text

        if r.status in (429, 500, 502, 503, 504):
            raise RetryableAPIError(f"Retryable API error {r.status}: {response_text}")  # noqa
        raise Exception(f"Non-retryable API error {r.status}: {response_text}")  # noqa


# ---------- Processing Loop ----------


async def process_docs(docs: list[dict[str, Any]], config: dict[str, Any]) -> None:
    processed = load_processed_ids(get_output_path_for_config(ANNOTATION_FILE, config))
    sem = asyncio.Semaphore(config["concurrent_requests"])
    output_lock = asyncio.Lock()
    fail_lock = asyncio.Lock()

    async with aiohttp.ClientSession() as session:

        async def worker(doc: dict[str, Any]) -> None:
            if "task_id" not in doc or "html" not in doc:
                logger.warning(f"Skipping invalid document: {doc}: missing 'task_id' or 'html' fields.")  # noqa
                return
            if doc["task_id"] in processed:
                logger.info(f"Skipping task_id {doc['task_id']} (already processed)")  # noqa
                return

            async with sem:
                start_time = time.perf_counter()
                try:
                    result_text = await call_llm(session, doc, config)
                    annotations = extract_annotations(result_text)
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    duration_ms = round(duration_ms)
                    timestamp = datetime.now(UTC).strftime("%d/%m/%Y %H:%M:%S")
                    result = {
                        "timestamp": timestamp,
                        "config": config["name"],
                        "model": config["model"],
                        "task_id": doc["task_id"],
                        "duration_ms": duration_ms,
                        "annotations": annotations,
                    }
                    async with output_lock:
                        save_jsonl_line(result, get_output_path_for_config(ANNOTATION_FILE, config))
                        processed.add(doc["task_id"])
                    logger.info(f"Processed {doc['task_id']}")  # noqa
                except Exception as e:  # noqa BLE001
                    timestamp = datetime.now(UTC).strftime("%d/%m/%Y %H:%M:%S")
                    traceback_text = traceback.format_exc()
                    async with fail_lock:
                        save_jsonl_line(
                            {
                                "timestamp": timestamp,
                                "config": config["name"],
                                "model": config["model"],
                                "task_id": doc["task_id"],
                                "error": f"{e}\n{traceback_text}",
                            },
                            get_output_path_for_config(FAILURES_FILE, config),
                        )

                    logger.error(f"Failed {doc.get('task_id')}: {e}\n{traceback_text}")  # noqa

        await asyncio.gather(*(worker(doc) for doc in docs))


# ---------- Entry Point ----------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate main content in a given HTML file collection using an LLM")

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input file (.jsonl or .parquet).",
    )

    parser.add_argument("--config", "-c", required=True, help="Path to config file.")

    parser.add_argument(
        "--max_docs",
        "-m",
        type=int,
        default=None,
        help="Caps the number of documents to be processed. Useful for debugging.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    docs = load_docs(args.input)[: args.max_docs]
    logger.info(f"Loaded {len(docs)} documents.")  # noqa G004
    asyncio.run(process_docs(docs, config))
    logger.info("Processing complete.")
