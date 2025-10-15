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
        self._validate_extension()

    def _validate_extension(self) -> None:
        if self.file_ext not in self.SUPPORTED_FORMATS:
            msg = f"Unsupported file format: {self.file_ext}. Supported: {list(self.SUPPORTED_FORMATS.keys())}"
            self.logger.error(msg)
            raise ValueError(msg)

    def to_polars(self, **polars_kwargs: Any) -> polars.DataFrame:  # noqa: ANN401
        """Load file into Polars DataFrame with optional kwargs for the reader."""
        self.logger.info("Loading %s into Polars DataFrame from %s", self.file_ext, self.file_path)

        match self.file_ext:
            case ".parquet":
                dataframe = polars.read_parquet(source=self.file_path, **polars_kwargs)
            case ".json":
                dataframe = polars.read_json(source=self.file_path, **polars_kwargs)
            case _:
                msg = f"Unsupported file format: {self.file_ext}. Supported: {list(self.SUPPORTED_FORMATS.keys())}"
                self.logger.error(msg)
                raise ValueError(msg)

        self.logger.info("Loaded DataFrame with shape %s", dataframe.shape)
        return dataframe

    def to_dataset(self, **dataset_kwargs: Any) -> datasets.Dataset:  # noqa: ANN401
        """Load file into Hugging Face Dataset with optional kwargs for load_dataset."""
        self.logger.info("Loading %s into Hugging Face Dataset from %s", self.file_ext, self.file_path)

        # extract split if provided, default to "test"
        split = dataset_kwargs.pop("split", "test")

        format_type = self.SUPPORTED_FORMATS[self.file_ext]
        dataset = datasets.load_dataset(
            format_type,
            data_files={split: str(self.file_path)},
            split=split,
            **dataset_kwargs,
        )

        self.logger.info("Loaded Dataset with length %d", len(dataset))
        return dataset


class HuggingFaceDataSource(BaseDataSource):
    """Represents a data source from the Hugging Face Hub."""

    def __init__(self, path: str, name: str | None = None) -> None:
        super().__init__()
        self.path = path
        self.name = name
        self.logger.info("Preparing to load from Hugging Face Hub: %s (name=%s)", path, name)

    def to_dataset(self, **dataset_kwargs: Any) -> datasets.Dataset:  # noqa: ANN401
        """Load from Hugging Face Hub into Dataset."""
        self.logger.info("Loading into Hugging Face Dataset...")

        # build load_dataset arguments
        load_args = {"path": self.path}
        if self.name:
            load_args["name"] = self.name

        # merge with user kwargs (user kwargs take precedence)
        load_args.update(dataset_kwargs)

        dataset = datasets.load_dataset(**load_args)

        # handle DatasetDict case
        if isinstance(dataset, datasets.DatasetDict):
            # check if split was specified in kwargs
            if "split" in dataset_kwargs and dataset_kwargs["split"] in dataset:
                selected_split = dataset_kwargs["split"]
            else:
                selected_split = next(iter(dataset.keys()))
                self.logger.warning(
                    "No split specified or split not found, automatically selecting first split: '%s'", selected_split
                )
            dataset = dataset[selected_split]

        elif not isinstance(dataset, datasets.Dataset):
            msg = f"Expected a Dataset or DatasetDict, but got {type(dataset)}"
            self.logger.error(msg)
            raise TypeError(msg)

        self.logger.info("Loaded Dataset with length %d", len(dataset))
        return dataset

    def to_polars(
        self,
        dataset_kwargs: dict[str, Any] | None = None,
        **polars_kwargs: Any,  # noqa: ANN401
    ) -> polars.DataFrame:
        """Load from Hugging Face Hub into Polars DataFrame."""
        self.logger.info("Loading into Polars DataFrame via Hugging Face Dataset...")

        dataset_kwargs = dataset_kwargs or {}
        dataset = self.to_dataset(**dataset_kwargs)
        dataframe = dataset.to_polars(**polars_kwargs)
        self.logger.info("Converted to DataFrame with shape %s", dataframe.shape)
        return dataframe


class D2GDataLoader:
    """Factory class to create data sources."""

    @staticmethod
    def from_file(file_path: str | pathlib.Path) -> FileDataSource:
        """Create a FileDataSource from a local file path."""
        return FileDataSource(file_path)

    @staticmethod
    def from_huggingface(path: str, name: str | None = None) -> HuggingFaceDataSource:
        """
        Create a HuggingFaceDataSource from Hugging Face Hub.

        Args:
            path: Dataset path on Hugging Face Hub
            name: Dataset configuration name (optional)
        """
        return HuggingFaceDataSource(path, name=name)
