from __future__ import annotations

import logging
import pathlib
from typing import Any, ClassVar

import polars  # noqa: ICN001

polars.Config.set_tbl_rows(-1)  # Show all rows
polars.Config.set_tbl_cols(-1)  # Show all columns
polars.Config.set_fmt_str_lengths(100)  # Set max string length for display


class TimeStats:
    LEAD_TIME_COLUMN: ClassVar[str] = "lead_time"

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def _round_numeric_value(self, value: Any, round_val: int, value_name: str = "value") -> float | None:  # noqa: ANN401
        """Round numeric values, handle None, and validate types."""
        if isinstance(value, (int, float)):
            return round(float(value), round_val)
        elif value is None:  # noqa: RET505
            return None
        else:
            msg = f"Unexpected type {type(value)} for {value_name}: {value}"
            self.logger.error(msg)
            raise TypeError(msg)

    def calculate_time_by(
        self,
        *,
        dataframe: polars.DataFrame,
        grouping_column: str,
        round_val: int = 3,
        save_dir: str | pathlib.Path = "ls_snapshot_output",
    ) -> tuple[polars.DataFrame, polars.DataFrame]:
        self.logger.info(
            "Calculating stats for %s! Grouping dataframe by %s...", self.LEAD_TIME_COLUMN, grouping_column
        )

        group_df = dataframe.group_by(grouping_column)
        time_stats = []

        for group_identifier, group in group_df:
            self.logger.debug("Grouped by (%s): %s", grouping_column, group_identifier[0])
            self.logger.debug("Group height: %d", group.height)

            filtered_group = group.filter(polars.col(self.LEAD_TIME_COLUMN).is_not_null())
            if filtered_group.is_empty():
                self.logger.warning(
                    "Group (%s: %s) is empty after filtering out None values in %s.",
                    grouping_column,
                    group_identifier[0],
                    self.LEAD_TIME_COLUMN,
                )
                msg = f"(Impossible?) Something went wrong, group ({grouping_column}: {group_identifier[0]}) is empty after filtering out None values in {self.LEAD_TIME_COLUMN}."
                raise ValueError(msg)

            self.logger.debug("Filtered group size (without None): %s", filtered_group.shape)
            mean_val = filtered_group[self.LEAD_TIME_COLUMN].mean()
            median_val = filtered_group[self.LEAD_TIME_COLUMN].median()
            max_val = filtered_group[self.LEAD_TIME_COLUMN].max()
            min_val = filtered_group[self.LEAD_TIME_COLUMN].min()

            mean_val = self._round_numeric_value(mean_val, round_val, "mean")
            median_val = self._round_numeric_value(median_val, round_val, "median")
            max_val = self._round_numeric_value(max_val, round_val, "max")
            min_val = self._round_numeric_value(min_val, round_val, "min")

            self.logger.debug("Mean lead time: %d", mean_val)
            self.logger.debug("Median lead time: %d", median_val)
            self.logger.debug("Max lead time: %d", max_val)
            self.logger.debug("Min lead time: %d", min_val)

            time_stats.append(
                {
                    grouping_column: group_identifier[0],
                    "annotations": group.height,
                    "size": filtered_group.height,
                    "mean": mean_val,
                    "median": median_val,
                    "max": max_val,
                    "min": min_val,
                }
            )
        df_with_time_stats = polars.DataFrame(time_stats).sort(grouping_column)
        self.logger.info("Calculated time statistics for %d groups.", df_with_time_stats.height)

        overall_mean = self._round_numeric_value(df_with_time_stats["mean"].mean(), round_val, "overall_mean")
        overall_median = self._round_numeric_value(df_with_time_stats["median"].median(), round_val, "overall_median")
        overall_max = self._round_numeric_value(df_with_time_stats["max"].max(), round_val, "overall_max")
        overall_min = self._round_numeric_value(df_with_time_stats["min"].min(), round_val, "overall_min")

        self.logger.info("Overall statistics across all groups:")
        self.logger.info("Overall mean of mean lead time: %s", overall_mean)
        self.logger.info("Overall median of median lead time: %s", overall_median)
        self.logger.info("Overall max of max lead time: %s", overall_max)
        self.logger.info("Overall min of min lead time: %s", overall_min)

        df_with_time_stats_with_overall = polars.DataFrame(
            {
                "annotations": df_with_time_stats["annotations"].sum(),
                "size": df_with_time_stats["size"].sum(),
                "mean": overall_mean,
                "median": overall_median,
                "max": overall_max,
                "min": overall_min,
            }
        )
        sub_savedir = pathlib.Path(save_dir).joinpath("stats")
        sub_savedir.mkdir(parents=True, exist_ok=True)

        path_df_with_time_stats = sub_savedir.joinpath(f"time_stats_{grouping_column}").with_suffix(".csv")
        df_with_time_stats.write_csv(path_df_with_time_stats)
        self.logger.info("Saved time statistics by %s to %s", grouping_column, path_df_with_time_stats)

        path_df_with_time_stats_overall = sub_savedir.joinpath(f"time_stats_overall_{grouping_column}").with_suffix(
            ".csv"
        )
        df_with_time_stats_with_overall.write_csv(path_df_with_time_stats_overall)
        self.logger.info("Saved overall time statistics by %s to %s", grouping_column, path_df_with_time_stats_overall)
        return df_with_time_stats, df_with_time_stats_with_overall
