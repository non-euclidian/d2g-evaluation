import pathlib

import polars  # noqa: ICN001

from labelstudio_snapshot.snapshot_preparation import SnapshotPreparationPipeline
from labelstudio_snapshot.stats.stats_k_alpha_questions import KrippendorffAlphaQuestions
from labelstudio_snapshot.stats.stats_self_confidence import SelfConfidenceStats
from labelstudio_snapshot.stats.stats_time import TimeStats


def run_prepare_snapshot(
    path_to_json: str | pathlib.Path, save_dir: str | pathlib.Path = "ls_snapshot_output/demo"
) -> tuple[polars.DataFrame, polars.DataFrame]:
    pipeline = SnapshotPreparationPipeline()

    df_per_row, df_per_task = pipeline.prepare_data(json_path=path_to_json, save_dir=save_dir)

    # calculate Krippendorff's alpha per question
    k_alpha_questions = KrippendorffAlphaQuestions()
    _, _ = k_alpha_questions.calculate_k_alpha(
        dataframe=df_per_row,
        use_columns=[
            "check_loading",
            "check_not_empty",
            "check_lang_match",
            "check_harmful_content",
            "check_highlightable",
        ],
        save_dir=save_dir,
    )

    # calculate self-confidence stats
    self_conf_stats = SelfConfidenceStats()
    _ = self_conf_stats.calculate_self_confidence(
        dataframe=df_per_row,
        grouping_column="task_id",
        save_dir=save_dir,
    )
    _ = self_conf_stats.calculate_self_confidence(
        dataframe=df_per_row,
        grouping_column="annotated_by",
        save_dir=save_dir,
    )

    # calculate time stats
    time_stats = TimeStats()
    _ = time_stats.calculate_time_by(
        dataframe=df_per_row,
        grouping_column="task_id",
        save_dir=save_dir,
    )
    _ = time_stats.calculate_time_by(
        dataframe=df_per_row,
        grouping_column="annotated_by",
        save_dir=save_dir,
    )

    return df_per_row, df_per_task
