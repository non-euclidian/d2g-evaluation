from __future__ import annotations

import logging
import pathlib
from typing import ClassVar

import polars  # noqa: ICN001

polars.Config.set_tbl_rows(-1)  # Show all rows
polars.Config.set_tbl_cols(-1)  # Show all columns
polars.Config.set_fmt_str_lengths(100)  # Set max string length for display


class TimeStats:
    LEAD_TIME_COLUMN: ClassVar[str] = "lead_time"

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def calculate_time_by(
        self,
        *,
        dataframe: polars.DataFrame,
        grouping_column: str,
        round_val: int | None = 2,
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

            time_stats.append(
                {
                    grouping_column: group_identifier[0],
                    "n_items": group.height,
                    "n_annotations": filtered_group.height,
                    "mean": mean_val,
                    "median": median_val,
                    "max": max_val,
                    "min": min_val,
                }
            )
        df_with_time_stats = polars.DataFrame(time_stats).sort(grouping_column)
        self.logger.info("Calculated time statistics for %d groups.", df_with_time_stats.height)

        overall_mean_of_mean = df_with_time_stats["mean"].mean()
        overall_mean_of_median = df_with_time_stats["median"].mean()

        overall_median_of_mean = df_with_time_stats["mean"].median()
        overall_median_of_median = df_with_time_stats["median"].median()

        overall_max = df_with_time_stats["max"].max()
        overall_min = df_with_time_stats["min"].min()

        df_with_time_stats_with_overall = polars.DataFrame(
            {
                "n_items": df_with_time_stats["n_items"].sum(),
                "n_annotations": df_with_time_stats["n_annotations"].sum(),
                "mean_of_mean": overall_mean_of_mean,
                "mean_of_median": overall_mean_of_median,
                "median_of_mean": overall_median_of_mean,
                "median_of_median": overall_median_of_median,
                "max": overall_max,
                "min": overall_min,
            }
        )
        if round_val is not None:
            df_with_time_stats_with_overall = df_with_time_stats_with_overall.with_columns(
                polars.all().round(round_val)
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
