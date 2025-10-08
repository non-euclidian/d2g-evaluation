import logging
import pathlib
from typing import ClassVar

import polars  # noqa: ICN001

polars.Config.set_tbl_rows(-1)  # Show all rows
polars.Config.set_tbl_cols(-1)  # Show all columns
polars.Config.set_fmt_str_lengths(100)  # Set max string length for display


class SelfConfidenceStats:
    CONFIDENCE_COLUMN: ClassVar[str] = "rate_conf_yourself"

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def calculate_self_confidence(
        self,
        *,
        dataframe: polars.DataFrame,
        grouping_column: str,
        round_val: int | None = 2,
        save_dir: str | pathlib.Path = "ls_snapshot_output",
    ) -> tuple[polars.DataFrame, polars.DataFrame]:
        self.logger.info(
            "Calculating stats for %s! Grouping dataframe by %s...", self.CONFIDENCE_COLUMN, grouping_column
        )

        group_df = dataframe.group_by(grouping_column)
        confidence_stats = []

        for group_identifier, group in group_df:
            self.logger.debug("Grouped by (%s): %s", grouping_column, group_identifier[0])
            self.logger.debug("Group size: %s", group.shape)

            # remove rows with None in CONFIDENCE_COLUMN
            filtered_group = group.filter(polars.col(self.CONFIDENCE_COLUMN).is_not_null())
            if filtered_group.is_empty():
                self.logger.warning(
                    "Group (%s: %s) is empty after filtering out None values in %s.",
                    grouping_column,
                    group_identifier[0],
                    self.CONFIDENCE_COLUMN,
                )
                confidence_stats.append(
                    {
                        grouping_column: group_identifier[0],
                        "n_items": group.height,
                        "n_annotations": 0,
                        "mean": None,
                        "median": None,
                        "max": None,
                        "min": None,
                    }
                )
                continue

            self.logger.debug("Filtered group size (without None): %s", filtered_group.shape)
            mean_val = filtered_group[self.CONFIDENCE_COLUMN].mean()
            median_val = filtered_group[self.CONFIDENCE_COLUMN].median()
            max_val = filtered_group[self.CONFIDENCE_COLUMN].max()
            min_val = filtered_group[self.CONFIDENCE_COLUMN].min()

            confidence_stats.append(
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
        df_with_confidence_stats = polars.DataFrame(confidence_stats).sort(grouping_column)
        self.logger.info("Calculated self-confidence statistics for %d groups.", df_with_confidence_stats.height)

        overall_mean_of_mean = df_with_confidence_stats["mean"].mean()
        overall_mean_of_median = df_with_confidence_stats["median"].mean()

        overall_median_of_mean = df_with_confidence_stats["mean"].median()
        overall_median_of_median = df_with_confidence_stats["median"].median()

        overall_max = df_with_confidence_stats["max"].max()
        overall_min = df_with_confidence_stats["min"].min()

        df_with_confidence_stats_with_overall = polars.DataFrame(
            {
                "n_items": df_with_confidence_stats["n_items"].sum(),
                "n_annotations": df_with_confidence_stats["n_annotations"].sum(),
                "mean_of_mean": overall_mean_of_mean,
                "mean_of_median": overall_mean_of_median,
                "median_of_mean": overall_median_of_mean,
                "median_of_median": overall_median_of_median,
                "max": overall_max,
                "min": overall_min,
            }
        )
        if round_val is not None:
            df_with_confidence_stats_with_overall = df_with_confidence_stats_with_overall.with_columns(
                polars.all().round(round_val)
            )

        sub_savedir = pathlib.Path(save_dir).joinpath("stats")
        sub_savedir.mkdir(parents=True, exist_ok=True)

        path_df_with_confidence_stats = sub_savedir.joinpath(f"self_confidence_{grouping_column}").with_suffix(".csv")
        df_with_confidence_stats.write_csv(path_df_with_confidence_stats)
        self.logger.info("Saved self-confidence statistics by %s to %s", grouping_column, path_df_with_confidence_stats)

        path_df_with_confidence_stats_with_overall = sub_savedir.joinpath(
            f"self_confidence_overall_{grouping_column}"
        ).with_suffix(".csv")
        df_with_confidence_stats_with_overall.write_csv(path_df_with_confidence_stats_with_overall)
        self.logger.info(
            "Saved overall self-confidence statistics by %s to %s",
            grouping_column,
            path_df_with_confidence_stats_with_overall,
        )

        return df_with_confidence_stats, df_with_confidence_stats_with_overall
