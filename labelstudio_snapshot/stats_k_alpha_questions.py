import logging
from dataclasses import dataclass
from enum import StrEnum, unique
from typing import ClassVar

import krippendorff  # type: ignore
import numpy  # noqa: ICN001
import polars  # noqa: ICN001

polars.Config.set_tbl_rows(-1)  # Show all rows
polars.Config.set_tbl_cols(-1)  # Show all columns
polars.Config.set_fmt_str_lengths(100)  # Set max string length for display


@unique
class LevelOfMeasurement(StrEnum):
    NOMINAL = "nominal"
    ORDINAL = "ordinal"
    INTERVAL = "interval"
    RATIO = "ratio"


@dataclass(frozen=True)
class KAlphaResult:
    """Result of Krippendorff's alpha calculation for a single group."""

    group_id: str | int
    size: int
    k_alpha: float

    @property
    def is_valid(self) -> bool:
        """Check if the result is valid (not NaN)."""
        return not numpy.isnan(self.k_alpha)


class KrippendorffAlphaQuestions:
    GROUPING_COLUMN: ClassVar[str] = "task_id"
    SEED: ClassVar[int] = 414242
    MIN_GROUP_SIZE: ClassVar[int] = 2
    MIN_COLUMNS: ClassVar[int] = 2
    SUPPORTED_LEVELS: ClassVar[set[str]] = {LevelOfMeasurement.NOMINAL}

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def calculate_k_alpha(
        self,
        *,
        dataframe: polars.DataFrame,
        use_columns: list[str],
        level_of_measurement: LevelOfMeasurement = LevelOfMeasurement.NOMINAL,
        limit_group: int | None = None,
    ) -> polars.DataFrame:
        self._validate_inputs(dataframe, use_columns, level_of_measurement)

        self.logger.info("Calculating Krippendorff's alpha with level: %s", level_of_measurement)
        self.logger.info("Input shape: %s, grouped by: %s", dataframe.shape, self.GROUPING_COLUMN)
        self.logger.info("Columns: %s", list(use_columns))

        results = self._process_groups(dataframe, use_columns, level_of_measurement, limit_group)

        return self._results_to_dataframe(results)

    def _validate_inputs(
        self,
        dataframe: polars.DataFrame,
        use_columns: list[str],
        level_of_measurement: str,
    ) -> None:
        """Validate input parameters."""
        if level_of_measurement not in self.SUPPORTED_LEVELS:
            msg = f"Unsupported level: {level_of_measurement}. Supported: {', '.join(sorted(self.SUPPORTED_LEVELS))}"
            self.logger.error(msg)
            raise ValueError(msg)

        if len(use_columns) < self.MIN_COLUMNS:
            msg = f"At least {self.MIN_COLUMNS} columns required, got {len(use_columns)}"
            self.logger.error(msg)
            raise ValueError(msg)

        if self.GROUPING_COLUMN not in dataframe.columns:
            msg = f"Grouping column '{self.GROUPING_COLUMN}' not found"
            self.logger.error(msg)
            raise ValueError(msg)

        missing_cols = set(use_columns) - set(dataframe.columns)
        if missing_cols:
            msg = f"Columns not found: {missing_cols}"
            self.logger.error(msg)
            raise ValueError(msg)

    def _process_groups(
        self,
        dataframe: polars.DataFrame,
        use_columns: list[str],
        level_of_measurement: str,
        limit_group: int | None,
    ) -> list[KAlphaResult]:
        """Process each group and calculate alpha."""
        results = []
        grouped = dataframe.group_by(self.GROUPING_COLUMN, maintain_order=True)

        for idx, (group_id, group) in enumerate(grouped, start=1):
            result = self._process_single_group(
                group_id=group_id[0],
                group=group,
                use_columns=use_columns,
                level_of_measurement=level_of_measurement,
                limit_group=limit_group,
                group_idx=idx,
            )
            results.append(result)

        return results

    def _process_single_group(  # noqa: PLR0913
        self,
        *,
        group_id: str | int,
        group: polars.DataFrame,
        use_columns: list[str],
        level_of_measurement: str,
        limit_group: int | None,
        group_idx: int,
    ) -> KAlphaResult:
        """Process a single group and calculate its alpha value."""
        self.logger.info("Group %d (%s: %s)", group_idx, self.GROUPING_COLUMN, group_id)
        self.logger.debug("Group shape: %s", group.shape)

        # check minimum group size
        if group.height < self.MIN_GROUP_SIZE:
            self.logger.warning("Group size (%d) < minimum (%d). Skipping.", group.height, self.MIN_GROUP_SIZE)
            return KAlphaResult(group_id, group.height, float("nan"))

        # apply sampling limit if specified
        group = self._apply_limit(group, limit_group)

        # extract and validate data
        data = group.select(use_columns).to_numpy()
        self.logger.debug("Data shape: %s", data.shape)

        # handle edge case: all values are identical
        if self._has_single_unique_value(data):
            self.logger.warning("Group %s has only one unique value. Setting alpha=1.0", group_id)
            return KAlphaResult(group_id, group.height, 1.0)

        # calculate alpha
        alpha = krippendorff.alpha(reliability_data=data, level_of_measurement=level_of_measurement)
        self.logger.info("Krippendorff's alpha: %.4f", alpha)
        return KAlphaResult(group_id, group.height, float(alpha))

    def _apply_limit(self, group: polars.DataFrame, limit_group: int | None) -> polars.DataFrame:
        """Apply sampling limit to group if specified."""
        if limit_group is None:
            return group

        max_limit = group.height
        if limit_group > max_limit:
            self.logger.warning("Limit (%d) > group size (%d). Using full group.", limit_group, max_limit)
            return group

        self.logger.debug("Sampling %d rows from group", limit_group)
        return group.sample(limit_group, seed=self.SEED)

    @staticmethod
    def _has_single_unique_value(data: numpy.ndarray) -> bool:
        """Check if data contains only one unique non-NaN value."""
        flat_data = data.flatten()
        non_nan_data = flat_data[~numpy.isnan(flat_data)]
        return len(numpy.unique(non_nan_data)) == 1

    def _results_to_dataframe(self, results: list[KAlphaResult]) -> polars.DataFrame:
        """Convert results to a Polars DataFrame."""
        data = [
            {
                self.GROUPING_COLUMN: r.group_id,
                "size": r.size,
                "k_alpha": r.k_alpha,
            }
            for r in results
        ]
        return polars.DataFrame(data).sort(self.GROUPING_COLUMN)
