import logging
import pathlib
from typing import Any, ClassVar

import polars  # noqa: ICN001

polars.Config.set_tbl_rows(-1)  # Show all rows
polars.Config.set_tbl_cols(-1)  # Show all columns
polars.Config.set_fmt_str_lengths(100)  # Set max string length for display


class SelfConfidenceStats:
    CONFIDENCE_COLUMN: ClassVar[str] = "rate_conf_yourself"

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

    def calculate_self_confidence(
        self,
        *,
        dataframe: polars.DataFrame,
        grouping_column: str,
        round_val: int = 3,
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
                        "annotations": group.height,
                        "size": 0,
                        "mean": None,
                        "median": None,
                        "max": None,
                        "min": None,
                    }
                )
                continue

            mean_val = filtered_group[self.CONFIDENCE_COLUMN].mean()
            median_val = filtered_group[self.CONFIDENCE_COLUMN].median()
            max_val = filtered_group[self.CONFIDENCE_COLUMN].max()
            min_val = filtered_group[self.CONFIDENCE_COLUMN].min()

            mean_val = self._round_numeric_value(mean_val, round_val, "mean_val")
            median_val = self._round_numeric_value(median_val, round_val, "median_val")
            max_val = self._round_numeric_value(max_val, round_val, "max_val")
            min_val = self._round_numeric_value(min_val, round_val, "min_val")

            self.logger.debug("Mean self-confidence: %d", mean_val)
            self.logger.debug("Median self-confidence: %c", median_val)
            self.logger.debug("Max self-confidence: %s", max_val)
            self.logger.debug("Min self-confidence: %s", min_val)

            confidence_stats.append(
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
        df_with_confidence_stats = polars.DataFrame(confidence_stats).sort(grouping_column)
        self.logger.info("Calculated self-confidence statistics for %d groups.", df_with_confidence_stats.height)

        overall_mean = df_with_confidence_stats["mean"].mean()
        overall_median = df_with_confidence_stats["median"].median()
        overall_max = df_with_confidence_stats["max"].max()
        overall_min = df_with_confidence_stats["min"].min()

        overall_mean = self._round_numeric_value(overall_mean, round_val, "overall_mean")
        overall_median = self._round_numeric_value(overall_median, round_val, "overall_median")
        overall_max = self._round_numeric_value(overall_max, round_val, "overall_max")
        overall_min = self._round_numeric_value(overall_min, round_val, "overall_min")

        self.logger.info("Overall statistics across all groups:")
        self.logger.info("Overall mean of self-confidence: %s", overall_mean)
        self.logger.info("Overall median of self-confidence: %s", overall_median)
        self.logger.info("Overall max of self-confidence: %s", overall_max)
        self.logger.info("Overall min of self-confidence: %s", overall_min)

        df_with_confidence_stats_with_overall = polars.DataFrame(
            {
                "annotations": df_with_confidence_stats["annotations"].sum(),
                "size": df_with_confidence_stats["size"].sum(),
                "mean": overall_mean,
                "median": overall_median,
                "max": overall_max,
                "min": overall_min,
            }
        )
        sub_savedir = pathlib.Path(save_dir).joinpath("stats")
        sub_savedir.mkdir(parents=True, exist_ok=True)

        path_df_with_confidence_stats = sub_savedir.joinpath(f"self_confidence_{grouping_column}").with_suffix(".csv")
        df_with_confidence_stats.write_csv(path_df_with_confidence_stats)
        self.logger.info("Saved self-confidence stats per group to %s", path_df_with_confidence_stats)

        path_df_with_confidence_stats_with_overall = sub_savedir.joinpath(
            f"self_confidence_overall_{grouping_column}"
        ).with_suffix(".csv")
        df_with_confidence_stats_with_overall.write_csv(path_df_with_confidence_stats_with_overall)
        self.logger.info("Saved self-confidence stats with overall to %s", path_df_with_confidence_stats_with_overall)

        return df_with_confidence_stats, df_with_confidence_stats_with_overall
