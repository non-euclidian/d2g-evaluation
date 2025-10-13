import logging
import pathlib
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import datasets  # type: ignore
import polars  # noqa: ICN001


class BaseDataSource(ABC):
    """Abstract base class for a data source (e.g., file, Hugging Face Hub)."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def to_polars(self) -> polars.DataFrame: ...

    """Load the data source into a Polars DataFrame."""

    @abstractmethod
    def to_dataset(self) -> datasets.Dataset: ...

    """Load the data source into a Hugging Face Dataset."""


class FileDataSource(BaseDataSource):
    """Represents a data source from a local file path."""

    SUPPORTED_FORMATS: ClassVar[dict[str, str]] = {
        ".parquet": "parquet",
        ".json": "json",
    }

    def __init__(self, file_path: str | pathlib.Path) -> None:
        super().__init__()
        if not isinstance(file_path, pathlib.Path):
            file_path = pathlib.Path(file_path)

        if not file_path.exists():
            msg = f"File not found at path: {file_path}"
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        self.file_path = file_path
        self.file_ext = file_path.suffix

        if self.file_ext not in self.SUPPORTED_FORMATS:
            msg = f"Unsupported file format: {self.file_ext}. Supported: {list(self.SUPPORTED_FORMATS.keys())}"
            self.logger.error(msg)
            raise ValueError(msg)

    def to_polars(self) -> polars.DataFrame:
        self.logger.info("Loading %s into Polars DataFrame from %s", self.file_ext, self.file_path)

        match self.file_ext:
            case ".parquet":
                dataframe = polars.read_parquet(self.file_path)
            case ".json":
                dataframe = polars.read_json(self.file_path)
            case _:
                msg = f"Unsupported file format: {self.file_ext}. Supported: {list(self.SUPPORTED_FORMATS.keys())}"
                self.logger.error(msg)
                raise ValueError(msg)

        self.logger.info("Loaded DataFrame with shape %s", dataframe.shape)
        return dataframe

    def to_dataset(self, split: str = "test") -> datasets.Dataset:
        self.logger.info("Loading %s into Hugging Face Dataset from %s", self.file_ext, self.file_path)

        match self.file_ext:
            case ".parquet":
                dataset = datasets.load_dataset("parquet", data_files={split: str(self.file_path)})[split]
            case ".json":
                dataset = datasets.load_dataset("json", data_files={split: str(self.file_path)})[split]
            case _:
                msg = f"Unsupported file format: {self.file_ext}. Supported: {list(self.SUPPORTED_FORMATS.keys())}"
                self.logger.error(msg)
                raise ValueError(msg)

        self.logger.info("Loaded Dataset with length %d", len(dataset))
        return dataset


class HuggingFaceDataSource(BaseDataSource):
    """Represents a data source from the Hugging Face Hub."""

    def __init__(self, path: str, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__()
        self.path = path
        self.args = args
        self.kwargs = kwargs
        self.logger.info("Preparing to load from Hugging Face Hub: %s with args %s", path, kwargs)

    def to_dataset(self) -> datasets.Dataset:
        self.logger.info("Loading into Hugging Face Dataset...")
        dataset = datasets.load_dataset(self.path, *self.args, **self.kwargs)
        if not isinstance(dataset, datasets.Dataset):
            # load_dataset can return a DatasetDict, handle this gracefully.
            # here, we'll arbitrarily pick the first split if the user didn't specify one.
            if isinstance(dataset, datasets.DatasetDict):
                split_name = next(iter(dataset.keys()))
                self.logger.warning("No split specified, automatically selecting first split: '%s'", split_name)
                dataset = dataset[split_name]
            else:
                msg = f"Expected a Dataset or DatasetDict, but got {type(dataset)}"
                self.logger.error(msg)
                raise TypeError(msg)

    def to_polars(self) -> polars.DataFrame:
        self.logger.info("Loading into Polars DataFrame via Hugging Face Dataset...")
        dataset = self.to_dataset()
        dataframe = dataset.to_polars()
        self.logger.info("Converted to DataFrame with shape %s", dataframe.shape)
        return dataframe


class DataLoader:
    """Factory class to create data sources."""

    @staticmethod
    def from_file(file_path: str | pathlib.Path) -> FileDataSource:
        return FileDataSource(file_path)

    @staticmethod
    def from_huggingface(path: str, *args: Any, **kwargs: Any) -> HuggingFaceDataSource:  # noqa: ANN401
        return HuggingFaceDataSource(path, *args, **kwargs)
