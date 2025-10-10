import logging
import pathlib
from abc import ABC, abstractmethod  # noqa: F401
from dataclasses import dataclass
from typing import ClassVar

import polars  # noqa: ICN001


@dataclass(slots=True)
class TaskGroup:
    idx: int
    task_id: int
    group: polars.DataFrame


class BaseEval(ABC):  # noqa: B024
    SEED: ClassVar[int] = 414242

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def load_dataframe_from_file_path(self, file_path: str | pathlib.Path) -> polars.DataFrame:
        if not isinstance(file_path, pathlib.Path):
            file_path = pathlib.Path(file_path)
        self.logger.info("Loading data from %s", file_path)

        file_ext = file_path.suffix.lower()

        if file_ext == ".parquet":
            dataframe = polars.read_parquet(file_path)
        elif file_ext == ".json":
            dataframe = polars.read_json(file_path)
        else:
            msg = f"Unsupported file format: {file_ext}. Only .parquet and .json are supported."
            self.logger.error(msg)
            raise ValueError(msg)

        self.logger.info("Loaded data from %s with shape %s", file_path, dataframe.shape)
        return dataframe

    def load_dataframe_from_huggingface(self) -> polars.DataFrame:
        msg = "Not implemented yet."
        self.logger.error(msg)
        raise NotImplementedError(msg)

    def load_dataframe_from_huggingface_dataset(self) -> polars.DataFrame:
        msg = "Not implemented yet."
        self.logger.error(msg)
        raise NotImplementedError(msg)
