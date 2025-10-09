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

    def load_prepared_data(self, prepared_data_path: str | pathlib.Path) -> list[TaskGroup]:
        if not isinstance(prepared_data_path, pathlib.Path):
            prepared_data_path = pathlib.Path(prepared_data_path)

        file_ext = prepared_data_path.suffix.lower()

        if file_ext == ".parquet":
            dataframe = polars.read_parquet(prepared_data_path)
        elif file_ext == ".json":
            dataframe = polars.read_json(prepared_data_path)
        else:
            msg = f"Unsupported file format: {file_ext}. Only .parquet and .json are supported."
            self.logger.error(msg)
            raise ValueError(msg)

        self.logger.info("Loaded prepared data from %s with shape %s", prepared_data_path, dataframe.shape)

        dataframe_grouped = dataframe.group_by("task_id", maintain_order=True)

        all_task_groups = [
            TaskGroup(idx=idx, task_id=task_id[0], group=group)
            for idx, (task_id, group) in enumerate(dataframe_grouped, start=1)
        ]
        self.logger.info("Total task groups loaded: %d", len(all_task_groups))
        return all_task_groups
