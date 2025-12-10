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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiohttp
import polars as pl
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from llm.connectors import LLMConnector, OpenRouterConnector, RetryableAPIError

# ============ Configuration ============
API_KEY = str(os.getenv("LLM_API_KEY"))
ANNOTATION_FILE = "results.jsonl"
FAILURES_FILE = "failures.jsonl"
RUN_CONFIG = "config.json"
CONCURRENT_REQUESTS_DEFAULT = 5
MAX_RETRIES = 5

# ======================================


# ---------- Logging ----------


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ---------- Helpers ----------


def load_config(config: str) -> dict[str, Any]:
    """
    Load reproducible run configuration from the provided string.
    Accepts either a dictionary or path to JSON configuration file.

    Expected config keys:
      - model (str)
      - connector (str), e.g. 'openrouter'
      - prompt (dict), including 'system' and 'user' prompts

    Optional config keys:
      - concurrent_requests (int). Used to prevent rate limiting. The default value is set in CONCURRENT_REQUESTS

    Args:
        config (str): Dictionary containing configuration values or path to the JSON config file.

    Raises:
        ValueError: If required keys are missing or have wrong types.
    """
    config_autoname = None

    try:
        config = json.loads(config)
        if "model" in config and config["model"] and isinstance(config["model"], str):
            # keep last part of a name like 'deepseek/model-x1'
            config["model"] = config["model"].rstrip("/")  # edge case: trailing slash
            config_autoname = config["model"].split("/")[-1]
    except json.JSONDecodeError:
        pass

    if not isinstance(config, dict):
        p = Path(config)
        with p.open(encoding="utf-8") as f:
            config = json.load(f)
            config_autoname = p.name.replace(".config", "")

    schema = {
        "model": str,
        "connector": str,
        "concurrent_requests": int,
        "prompt": dict,
    }

    if "concurrent_requests" not in config:
        config["concurrent_requests"] = CONCURRENT_REQUESTS_DEFAULT

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

    if "name" not in config or not config["name"]:
        logger.warning(f"'name' missing from config. Using {config_autoname} as experiment name instead. ")  # noqa G004
        config["name"] = config_autoname
    logger.info(f"Loaded config {config}")  # noqa G004

    return config


def get_output_path_for_config(filename: str, config: dict[str, Any]) -> str:
    return_value = Path("llm/.annotations") / config["name"] / filename
    # create missing directories if they don't exist yet.
    # if they exist, do nothing.
    return_value.parent.mkdir(parents=True, exist_ok=True)

    return return_value


def load_docs(path: str) -> list[dict[str, Any]]:
    logger.info(f"Loading dataset from {path}")  # noqa G004
    p = Path(path)
    if p.suffix == ".parquet":
        df = pl.read_parquet(path)
        return df.to_dicts()
    # Assume JSONL
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_processed_ids(path: str) -> set[str]:
    """Return task_ids already processed successfully, so that they can be skipped."""
    logger.info(f"Loading existing LLM annotations from {path}")  # noqa G004
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


def get_connector_from_config(config: dict[str, Any]) -> LLMConnector:
    connector_name = config.get("connector")
    if connector_name is None:
        raise ValueError("Missing 'connector' in config.")  # noqa

    supported_providers = {
        "openrouter": OpenRouterConnector,
    }

    connector_cls = supported_providers.get(connector_name)
    if connector_cls is None:
        raise ValueError(f"Unsupported connector: {connector_name}")  # noqa

    return connector_cls(api_key=API_KEY, config=config)


# ---------- API Call ----------


@retry(
    wait=wait_exponential(multiplier=3, min=5, max=30),
    stop=stop_after_attempt(MAX_RETRIES),
    retry=retry_if_exception_type(RetryableAPIError),
)
async def call_llm(
    connector: LLMConnector,
    session: aiohttp.ClientSession,
    doc: dict[str, Any],
) -> dict[str, Any]:
    return await connector.call_llm(session, doc)


# ---------- Processing Loop ----------


async def process_docs(docs: list[dict[str, Any]], config: dict[str, Any]) -> None:
    processed = load_processed_ids(get_output_path_for_config(ANNOTATION_FILE, config))
    sem = asyncio.Semaphore(config["concurrent_requests"])
    output_lock = asyncio.Lock()
    fail_lock = asyncio.Lock()
    connector = get_connector_from_config(config)

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
                    parsed_response = await call_llm(connector, session, doc)
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    duration_ms = round(duration_ms)
                    timestamp = datetime.now(UTC).strftime("%d/%m/%Y %H:%M:%S")
                    result = {
                        "timestamp": timestamp,
                        "config": config["name"],
                        "model": config["model"],
                        "task_id": doc["task_id"],
                        "duration_ms": duration_ms,
                        "annotations": parsed_response["annotations"],
                        "total_tokens": parsed_response["total_tokens"],
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
    logger.info(f"Loaded {len(docs)} documents")  # noqa G004
    # copy config to experimental run directory for reproducibility
    run_config_path = get_output_path_for_config(RUN_CONFIG, config)
    with Path.open(run_config_path, "w") as f:
        json.dump(config, f, indent=2)
    asyncio.run(process_docs(docs, config))
    logger.info("Processing complete.")
