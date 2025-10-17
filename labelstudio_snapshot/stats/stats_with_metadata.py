import logging
import pathlib

import polars  # noqa: ICN001

polars.Config.set_tbl_rows(-1)  # Show all rows
polars.Config.set_tbl_cols(-1)  # Show all columns
polars.Config.set_fmt_str_lengths(100)  # Set max string length for display


class MetadataStats:
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def calculate_stats_with_metadata(
        self,
        *,
        dataframe: polars.DataFrame,
        save_dir: str | pathlib.Path = "ls_snapshot_output",
    ) -> None:
        if not isinstance(save_dir, pathlib.Path):
            save_dir = pathlib.Path(save_dir)

        sub_savedir = pathlib.Path(save_dir).joinpath("stats")
        sub_savedir.mkdir(parents=True, exist_ok=True)
        dataframe = dataframe.with_columns(year=polars.col("timestamp").dt.year())

        _ = self._calculate_core(dataframe=dataframe, save_dir=sub_savedir, round_val=1)
        df_collection = self._prepare_stats_by(  # noqa: F841
            dataframe=dataframe, column="collection", name="collection", save_dir=sub_savedir
        )
        df_year = self._prepare_stats_by(  # noqa: F841
            dataframe=dataframe, column="year", name="year", save_dir=sub_savedir
        )

    def _calculate_core(
        self, dataframe: polars.DataFrame, save_dir: pathlib.Path, round_val: int = 1
    ) -> polars.DataFrame:
        total_unique_tasks = dataframe["task_id"].n_unique()
        total_annotations = dataframe.height

        multi_number = 2

        single_span_tasks = dataframe.filter(polars.col("len_merged_spans") >= 1)["task_id"].n_unique()
        single_span_annotations = dataframe.filter(polars.col("len_merged_spans") >= 1).height

        # FIX: Count tasks with >= 2 valid annotations (not tasks with annotation having len_merged_spans >= 2)
        # This matches the evaluation logic in HumanVsHumanEvaluation
        task_annotation_counts = (
            dataframe.filter(polars.col("len_merged_spans") >= 1)
            .group_by("task_id")
            .agg(polars.len().alias("valid_annotation_count"))
        )
        tasks_with_multi_valid = task_annotation_counts.filter(polars.col("valid_annotation_count") >= multi_number)
        multi_span_tasks = tasks_with_multi_valid.shape[0]

        # Count annotations in tasks that have >= 2 valid annotations
        task_ids_with_multi = tasks_with_multi_valid["task_id"]
        multi_span_annotations = dataframe.filter(
            polars.col("task_id").is_in(task_ids_with_multi) & (polars.col("len_merged_spans") >= 1)
        ).height

        summary = polars.DataFrame(
            {
                "summary": [
                    "total tasks",
                    "total annotations",
                    "tasks with any text annotation",
                    "annotations with any text annotation",
                    f"tasks with >={multi_number} text annotations",
                    f"annotations with >={multi_number} text annotations",
                ],
                "result": [
                    total_unique_tasks,
                    total_annotations,
                    single_span_tasks,
                    single_span_annotations,
                    multi_span_tasks,
                    multi_span_annotations,
                ],
                "percentage": [
                    100.0,
                    100.0,
                    (single_span_tasks / total_unique_tasks * 100) if total_unique_tasks > 0 else 0.0,
                    (single_span_annotations / total_annotations * 100) if total_annotations > 0 else 0.0,
                    (multi_span_tasks / total_unique_tasks * 100) if total_unique_tasks > 0 else 0.0,
                    (multi_span_annotations / total_annotations * 100) if total_annotations > 0 else 0.0,
                ],
            }
        )

        summary = summary.with_columns(
            [
                polars.col("percentage").round(round_val).alias("percentage"),
                (polars.col("percentage").round(round_val).cast(polars.Utf8) + polars.lit("%")).alias(
                    "percentage_as_str"
                ),
            ]
        )

        path_file_summary = save_dir.joinpath("summary_stats").with_suffix(".csv")
        summary.write_csv(path_file_summary)

        return summary

    def _prepare_stats_by(
        self, dataframe: polars.DataFrame, column: str, name: str, save_dir: pathlib.Path
    ) -> polars.DataFrame:
        multi_number = 2

        # filter for single span (>= 1 valid annotation per task)
        single_span_df = dataframe.filter(polars.col("len_merged_spans") >= 1)

        # FIX: filter for multi span (tasks with >= 2 valid annotations)
        # First, identify tasks with >= 2 valid annotations
        task_annotation_counts = (
            dataframe.filter(polars.col("len_merged_spans") >= 1)
            .group_by("task_id")
            .agg(polars.len().alias("valid_annotation_count"))
        )
        tasks_with_multi_valid = task_annotation_counts.filter(polars.col("valid_annotation_count") >= multi_number)
        task_ids_with_multi = tasks_with_multi_valid["task_id"]

        # Then filter dataframe to only include annotations from those tasks (and with valid spans)
        multi_span_df = dataframe.filter(
            polars.col("task_id").is_in(task_ids_with_multi) & (polars.col("len_merged_spans") >= 1)
        )

        # ==================== ANNOTATIONS (rows) ====================
        # calculate stats for all data
        name_base_annotations_all = "total_annotations"
        result_all = (
            dataframe[column]
            .value_counts(name=name_base_annotations_all)
            .sort(column)
            .with_columns(
                (polars.col(name_base_annotations_all) / polars.col(name_base_annotations_all).sum() * 100)
                .round(1)
                .alias(f"percentage_{name_base_annotations_all}"),
                (
                    (polars.col(name_base_annotations_all) / polars.col(name_base_annotations_all).sum() * 100)
                    .round(1)
                    .cast(polars.Utf8)
                    + "%"
                ).alias(f"percentage_as_str_{name_base_annotations_all}"),
            )
        )

        name_base_annotations_single = "annotations_with_any_text"

        # calculate stats for single span annotations
        result_single = (
            single_span_df[column]
            .value_counts(name=name_base_annotations_single)
            .sort(column)
            .with_columns(
                (polars.col(name_base_annotations_single) / polars.col(name_base_annotations_single).sum() * 100)
                .round(1)
                .alias(f"percentage_{name_base_annotations_single}"),
                (
                    (polars.col(name_base_annotations_single) / polars.col(name_base_annotations_single).sum() * 100)
                    .round(1)
                    .cast(polars.Utf8)
                    + "%"
                ).alias(f"percentage_as_str_{name_base_annotations_single}"),
            )
        )

        name_base_annotations_multi = f"annotations_with_>={multi_number}_text"

        # calculate stats for multi span annotations
        result_multi = (
            multi_span_df[column]
            .value_counts(name=name_base_annotations_multi)
            .sort(column)
            .with_columns(
                (polars.col(name_base_annotations_multi) / polars.col(name_base_annotations_multi).sum() * 100)
                .round(1)
                .alias(f"percentage_{name_base_annotations_multi}"),
                (
                    (polars.col(name_base_annotations_multi) / polars.col(name_base_annotations_multi).sum() * 100)
                    .round(1)
                    .cast(polars.Utf8)
                    + "%"
                ).alias(f"percentage_as_str_{name_base_annotations_multi}"),
            )
        )

        # ==================== TASKS (unique task_id) ====================
        # calculate unique tasks for all data
        name_base_tasks_all = "total_tasks"
        result_tasks_all = (
            dataframe.group_by(column)
            .agg(polars.col("task_id").n_unique().alias(name_base_tasks_all))
            .sort(column)
            .with_columns(
                (polars.col(name_base_tasks_all) / polars.col(name_base_tasks_all).sum() * 100)
                .round(1)
                .alias(f"percentage_{name_base_tasks_all}"),
                (
                    (polars.col(name_base_tasks_all) / polars.col(name_base_tasks_all).sum() * 100)
                    .round(1)
                    .cast(polars.Utf8)
                    + "%"
                ).alias(f"percentage_as_str_{name_base_tasks_all}"),
            )
        )

        name_base_tasks_single = "tasks_with_any_text"
        # calculate unique tasks for single span
        result_tasks_single = (
            single_span_df.group_by(column)
            .agg(polars.col("task_id").n_unique().alias(name_base_tasks_single))
            .sort(column)
            .with_columns(
                (polars.col(name_base_tasks_single) / polars.col(name_base_tasks_single).sum() * 100)
                .round(1)
                .alias(f"percentage_{name_base_tasks_single}"),
                (
                    (polars.col(name_base_tasks_single) / polars.col(name_base_tasks_single).sum() * 100)
                    .round(1)
                    .cast(polars.Utf8)
                    + "%"
                ).alias(f"percentage_as_str_{name_base_tasks_single}"),
            )
        )

        name_base_tasks_multi = f"tasks_with_>={multi_number}_text"
        # calculate unique tasks for multi span
        result_tasks_multi = (
            multi_span_df.group_by(column)
            .agg(polars.col("task_id").n_unique().alias(name_base_tasks_multi))
            .sort(column)
            .with_columns(
                (polars.col(name_base_tasks_multi) / polars.col(name_base_tasks_multi).sum() * 100)
                .round(1)
                .alias(f"percentage_{name_base_tasks_multi}"),
                (
                    (polars.col(name_base_tasks_multi) / polars.col(name_base_tasks_multi).sum() * 100)
                    .round(1)
                    .cast(polars.Utf8)
                    + "%"
                ).alias(f"percentage_as_str_{name_base_tasks_multi}"),
            )
        )

        # combine all results
        result = (
            result_all.join(result_single, on=column, how="left")
            .join(result_multi, on=column, how="left")
            .join(result_tasks_all, on=column, how="left")
            .join(result_tasks_single, on=column, how="left")
            .join(result_tasks_multi, on=column, how="left")
            .fill_null(0)  # fill nulls for categories that don't appear in filtered data
            .pipe(
                lambda df_: df_.vstack(
                    polars.DataFrame(
                        {
                            column: [None],
                            name_base_annotations_all: [df_[name_base_annotations_all].sum()],
                            f"percentage_{name_base_annotations_all}": [100.0],
                            f"percentage_as_str_{name_base_annotations_all}": ["100.0%"],
                            name_base_annotations_single: [df_[name_base_annotations_single].sum()],
                            f"percentage_{name_base_annotations_single}": [100.0],
                            f"percentage_as_str_{name_base_annotations_single}": ["100.0%"],
                            name_base_annotations_multi: [df_[name_base_annotations_multi].sum()],
                            f"percentage_{name_base_annotations_multi}": [100.0],
                            f"percentage_as_str_{name_base_annotations_multi}": ["100.0%"],
                            name_base_tasks_all: [df_[name_base_tasks_all].sum()],
                            f"percentage_{name_base_tasks_all}": [100.0],
                            f"percentage_as_str_{name_base_tasks_all}": ["100.0%"],
                            name_base_tasks_single: [df_[name_base_tasks_single].sum()],
                            f"percentage_{name_base_tasks_single}": [100.0],
                            f"percentage_as_str_{name_base_tasks_single}": ["100.0%"],
                            name_base_tasks_multi: [df_[name_base_tasks_multi].sum()],
                            f"percentage_{name_base_tasks_multi}": [100.0],
                            f"percentage_as_str_{name_base_tasks_multi}": ["100.0%"],
                        }
                    ).cast(
                        {
                            name_base_annotations_all: polars.UInt32,
                            name_base_annotations_single: polars.UInt32,
                            name_base_annotations_multi: polars.UInt32,
                            name_base_tasks_all: polars.UInt32,
                            name_base_tasks_single: polars.UInt32,
                            name_base_tasks_multi: polars.UInt32,
                        }
                    )
                )
            )
        )

        file_name = f"cv_stats_{column}_{name}"
        file_path = save_dir.joinpath(file_name).with_suffix(".csv")
        result.write_csv(file_path)
        return result
