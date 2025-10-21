from __future__ import annotations

import json
import logging
import pathlib
from typing import ClassVar

import polars  # noqa: ICN001
import tqdm  # type: ignore

from labelstudio_snapshot.merge_overlap import SpanOverlapMerger
from labelstudio_snapshot.pseudonymize import SnapshotPseudonymizer
from labelstudio_snapshot.snapshot_dataclasses import Annotations, LabelStudioTask, ProcessedData, Result


class SnapshotPreparationPipeline:
    ANNOTATIONS_COLUMN: ClassVar[str] = "annotations"
    TASK_GROUPING_COLUMNS: ClassVar[list[str]] = ["task_id", "file_name", "html"]
    ANNOTATION_COUNT_COLUMN: ClassVar[str] = "annotation_count"

    AGGREGATED_ANNOTATION_FIELDS: ClassVar[list[str]] = [
        "annotation_id",
        "annotated_by",
        "check_loading",
        "check_not_empty",
        "check_lang_match",
        "check_harmful_content",
        "check_highlightable",
        "rate_conf_yourself",
        "original_spans",
        "len_original_spans",
        "merged_spans",
        "len_merged_spans",
        "lead_time",
        "as_string",
        "len_as_string",
        "as_spans",
        "len_as_spans",
    ]

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

        self.overlap_span_merger = SpanOverlapMerger()

        self.pseudonymizer = SnapshotPseudonymizer()

        self.email_mapping: dict[str, str] = {}
        self.language: str | None = None

    def _extract_language_from_filename(self, json_path: pathlib.Path) -> str:
        """Extract language code from filename pattern {lang}___{other name}.json"""
        filename = json_path.name
        if "___" in filename:
            lang = filename.split("___")[0]
            if lang:
                return lang
        return "unknown_language"

    def _determine_language(self, json_path: pathlib.Path, language: str | None = None) -> str:
        """Determine language from parameter or filename, defaulting to 'unknown_language'"""
        if language is not None:
            return language
        return self._extract_language_from_filename(json_path)

    def read_json(self, json_path: str | pathlib.Path) -> list[dict]:
        self.logger.info("Reading JSON from %s", json_path)
        with pathlib.Path(json_path).open("r", encoding="utf-8") as json_file:
            json_data = json.load(json_file)
        self.logger.info("JSON has %d tasks!", len(json_data))
        return json_data

    def create_structured_data(self, json_data: list[dict]) -> list[LabelStudioTask]:
        self.logger.info("Creating structured data from JSON!")
        structured_data = [LabelStudioTask(**task) for task in json_data]
        self.logger.info("Structured data created!")
        self.logger.info("Structured data has %d tasks!", len(structured_data))
        return structured_data

    def _process_annotations(self, task: LabelStudioTask, annotation: Annotations) -> ProcessedData | None:
        self.logger.debug(
            "Task ID: %d | Annotation ID: %d | Email: %s",
            task.id,
            annotation.id,
            annotation.completed_by.email,
        )

        if annotation.result:
            self.logger.debug("Result: %s", annotation.result)
            original_email = annotation.completed_by.email
            pseudonymized_email = self.pseudonymizer.pseudonymize_email(original_email)

            if original_email not in self.email_mapping:
                self.email_mapping[original_email] = pseudonymized_email

            # take only the spans
            original_spans = [r.value for r in annotation.result if r.from_name == "spans"]
            # merge overlapping spans (it's ordered)
            merged_overlapped_spans = self.overlap_span_merger.merge_overlapping_text_spans(original_spans)

            raw_string = SnapshotSpanStrategies.join_spans_to_single_string(merged_overlapped_spans)
            spans = SnapshotSpanStrategies.spans_to_list(merged_overlapped_spans)

            assert len(original_spans) >= len(merged_overlapped_spans), (
                "Original spans should be greater than or equal to merged spans"
            )
            assert len(merged_overlapped_spans) == len(spans), "Merged spans and spans should have the same length"

            if self.language is None:
                msg = "Language must be specified before processing annotations"
                raise ValueError(msg)

            return ProcessedData(
                language=self.language,
                task_id=task.id,
                file_name=task.file_upload,
                annotation_id=annotation.id,
                annotated_by=pseudonymized_email,
                check_loading=SnapshotFieldExtractor.extract_single_choice(annotation.result, "check_loading", [1, 2]),
                check_not_empty=SnapshotFieldExtractor.extract_single_choice(
                    annotation.result, "check_not_empty", [3, 4]
                ),
                check_lang_match=SnapshotFieldExtractor.extract_single_choice(
                    annotation.result, "check_lang_match", [5, 6]
                ),
                check_harmful_content=SnapshotFieldExtractor.extract_single_choice(
                    annotation.result, "check_harmful_content", [7, 8]
                ),
                check_highlightable=SnapshotFieldExtractor.extract_single_choice(
                    annotation.result, "check_highlightable", [9, 10]
                ),
                rate_conf_yourself=SnapshotFieldExtractor.extract_rate_conf(annotation.result, "rate_conf_yourself"),
                original_spans=original_spans,
                len_original_spans=len(original_spans),
                merged_spans=merged_overlapped_spans,
                len_merged_spans=len(merged_overlapped_spans),
                html=task.data["html"],
                lead_time=annotation.lead_time,
                as_string=raw_string,  # our main string
                len_as_string=len(raw_string),
                as_spans=spans,
                len_as_spans=len(spans),
            )

        return None

    def process_structured_data(self, structured_data: list[LabelStudioTask]) -> list[ProcessedData]:
        ready_to_use_data = []

        for idx_task, task in tqdm.tqdm(
            enumerate(structured_data, start=1), total=len(structured_data), desc="Processing tasks"
        ):
            for idx_annot, annotation in enumerate(task.annotations, start=1):
                msg_debug = f""">Processing task {idx_task}/{len(structured_data)}. Processing annotation {idx_annot}/{len(task.annotations)}"""
                self.logger.debug(msg_debug)
                processed_data = self._process_annotations(task, annotation)
                if processed_data:
                    ready_to_use_data.append(processed_data)
                else:
                    msg = f"No result found for task {task.id} | annotation {annotation.id}"
                    raise ValueError(msg)

        self.logger.info("Processed data has %d tasks!", len(structured_data))
        self.logger.info("Processed data has %d annotations!", len(ready_to_use_data))
        return ready_to_use_data

    def convert_processed_data_to_dataframe(self, processed_data: list[ProcessedData]) -> polars.DataFrame:
        self.logger.info("Converting processed data to DataFrame!")
        df = polars.DataFrame(processed_data)
        self.logger.info("Dataframe shape: %s", df.shape)
        return df

    def load_and_process_data(
        self,
        json_path: str | pathlib.Path,
        language: str | None = None,
    ) -> tuple[list[dict], list[LabelStudioTask], list[ProcessedData], polars.DataFrame]:
        self.language = self._determine_language(json_path=pathlib.Path(json_path), language=language)
        self.logger.info("Processing data with language: %s", self.language)

        json_data = self.read_json(json_path=json_path)
        structured_data = self.create_structured_data(json_data=json_data)
        processed_structured_data = self.process_structured_data(structured_data=structured_data)
        dataframe_annotation_per_row = self.convert_processed_data_to_dataframe(
            processed_data=processed_structured_data
        )

        return json_data, structured_data, processed_structured_data, dataframe_annotation_per_row

    def _save_email_mapping(self, save_dir_path: pathlib.Path) -> None:
        mapping_dataframe = polars.DataFrame(
            {
                "original_email": list(self.email_mapping.keys()),
                "pseudonymized_email": list(self.email_mapping.values()),
            }
        )
        mapping_csv_path = save_dir_path.joinpath("email_mapping").with_suffix(".csv")
        mapping_dataframe.write_csv(mapping_csv_path)
        self.logger.info("Email mapping saved to CSV at %s", mapping_csv_path)

        mapping_parquet_path = save_dir_path.joinpath("email_mapping").with_suffix(".parquet")
        mapping_dataframe.write_parquet(mapping_parquet_path)
        self.logger.info("Email mapping saved to Parquet at %s", mapping_parquet_path)

    def _group_by_task_and_aggregate(self, dataframe: polars.DataFrame) -> polars.DataFrame:
        self.logger.info("Grouping by %s and aggregating annotations!", self.TASK_GROUPING_COLUMNS)
        grouped = dataframe.group_by(self.TASK_GROUPING_COLUMNS, maintain_order=True).agg(
            [
                polars.struct(self.AGGREGATED_ANNOTATION_FIELDS).alias(self.ANNOTATIONS_COLUMN),
                polars.count().alias(self.ANNOTATION_COUNT_COLUMN),
            ]
        )
        self.logger.info("Grouped DataFrame shape: %s", grouped.shape)
        return grouped

    def prepare_data(
        self,
        json_path: str | pathlib.Path,
        save_dir: str | pathlib.Path = "ls_snapshot_output",
        language: str | None = None,
    ) -> tuple[polars.DataFrame, polars.DataFrame]:
        _, _, _, dataframe_annotation_per_row = self.load_and_process_data(json_path=json_path, language=language)

        save_dir_path = pathlib.Path(save_dir).joinpath("snapshot_prepared")
        save_dir_path.mkdir(parents=True, exist_ok=True)

        df_json_path = save_dir_path.joinpath("snapshot_prepared_annotation_per_row").with_suffix(".json")
        df_parquet_path = save_dir_path.joinpath("snapshot_prepared_annotation_per_row").with_suffix(".parquet")

        dataframe_annotation_per_row.write_json(df_json_path)
        self.logger.info("Dataframe (annotation per row) saved to JSON at %s", df_json_path)
        dataframe_annotation_per_row.write_parquet(df_parquet_path)
        self.logger.info("Dataframe (annotation per row) saved to Parquet at %s", df_parquet_path)

        self._save_email_mapping(save_dir_path=save_dir_path)

        dataframe_annotation_per_task = self._group_by_task_and_aggregate(dataframe_annotation_per_row)

        df_task_json_path = save_dir_path.joinpath("snapshot_prepared_annotation_per_task").with_suffix(".json")
        df_task_parquet_path = save_dir_path.joinpath("snapshot_prepared_annotation_per_task").with_suffix(".parquet")

        dataframe_annotation_per_task.write_json(df_task_json_path)
        self.logger.info("Dataframe (annotation per task) saved to JSON at %s", df_task_json_path)
        dataframe_annotation_per_task.write_parquet(df_task_parquet_path)
        self.logger.info("Dataframe (annotation per task) saved to Parquet at %s", df_task_parquet_path)

        return dataframe_annotation_per_row, dataframe_annotation_per_task

    def prepare_data_with_metadata_restoration(
        self,
        json_path: str | pathlib.Path,
        raw_samples_path: str | pathlib.Path,
        save_dir: str | pathlib.Path = "ls_snapshot_output",
    ) -> tuple[polars.DataFrame, polars.DataFrame]:
        df_per_row, df_per_task = self.prepare_data(json_path=json_path, save_dir=save_dir)

        df_per_row_with_metadata = SnapshotMetadataRestorer.restore_minimal_metadata(
            json_path=json_path,
            raw_samples_path=raw_samples_path,
            dataframe=df_per_row,
            save_dir=save_dir,
        )

        df_per_task_with_metadata = SnapshotMetadataRestorer.restore_minimal_metadata(
            json_path=json_path,
            raw_samples_path=raw_samples_path,
            dataframe=df_per_task,
            save_dir=save_dir,
        )
        # save dataframes with metadata
        save_dir_path = pathlib.Path(save_dir).joinpath("snapshot_prepared")

        df_row_metadata_json_path = save_dir_path.joinpath(
            "snapshot_prepared_annotation_per_row_with_metadata"
        ).with_suffix(".json")
        df_row_metadata_parquet_path = save_dir_path.joinpath(
            "snapshot_prepared_annotation_per_row_with_metadata"
        ).with_suffix(".parquet")
        df_per_row_with_metadata.write_json(df_row_metadata_json_path)
        df_per_row_with_metadata.write_parquet(df_row_metadata_parquet_path)

        df_task_metadata_json_path = save_dir_path.joinpath(
            "snapshot_prepared_annotation_per_task_with_metadata"
        ).with_suffix(".json")
        df_task_metadata_parquet_path = save_dir_path.joinpath(
            "snapshot_prepared_annotation_per_task_with_metadata"
        ).with_suffix(".parquet")
        df_per_task_with_metadata.write_json(df_task_metadata_json_path)
        df_per_task_with_metadata.write_parquet(df_task_metadata_parquet_path)

        return df_per_row_with_metadata, df_per_task_with_metadata


class SnapshotFieldExtractor:
    @staticmethod
    def extract_single_choice(result: list[Result], field_name: str, encode_values_to: list[int]) -> int | None:
        if len(encode_values_to) != 2:  # noqa: PLR2004
            msg = "encode_values_to must have 2 elements!"
            raise ValueError(msg)

        values = [r.value for r in result if r.from_name == field_name]
        if not values:
            return None
        choice = values[0]["choices"][0]
        return encode_values_to[0] if choice == "Yes" else encode_values_to[1] if choice == "No" else None

    @staticmethod
    def extract_rate_conf(result: list[Result], field_name: str) -> int | None:
        values = [r.value for r in result if r.from_name == field_name]
        return values[0]["rating"] if values else None


class SnapshotSpanStrategies:
    @staticmethod
    def join_spans_to_single_string(spans: list[dict], separator: str = " ") -> str:
        return separator.join([span["text"] for span in spans])

    @staticmethod
    def spans_to_list(spans: list[dict]) -> list[str]:
        return [span["text"] for span in spans]


class SnapshotMetadataRestorer:
    @staticmethod
    def restore_minimal_metadata(
        json_path: str | pathlib.Path,
        raw_samples_path: str | pathlib.Path,
        dataframe: polars.DataFrame,
        save_dir: str | pathlib.Path = "ls_snapshot_output",
    ) -> polars.DataFrame:
        if not isinstance(json_path, pathlib.Path):
            json_path = pathlib.Path(json_path)

        if not isinstance(raw_samples_path, pathlib.Path):
            raw_samples_path = pathlib.Path(raw_samples_path)

        if not isinstance(save_dir, pathlib.Path):
            save_dir = pathlib.Path(save_dir)

        save_dir_path = pathlib.Path(save_dir).joinpath("snapshot_prepared")
        save_dir_path.mkdir(parents=True, exist_ok=True)

        lang_code = json_path.name.split("___")[0]
        if not lang_code:
            msg = f"Cannot extract language code from filename: {json_path.name}"
            raise ValueError(msg)

        paths_to_raw_samples = list(raw_samples_path.rglob(f"{lang_code}/**/*.parquet"))
        if not paths_to_raw_samples:
            msg = f"No raw sample files found for language code: {lang_code} in {raw_samples_path}"
            raise ValueError(msg)

        raw_samples_df = polars.read_parquet(
            paths_to_raw_samples, columns=["filename_warc", "url", "timestamp", "filename_html"]
        )

        # print("Sample file_name values:")
        # print(dataframe.select("file_name").head())
        # print("\nExtracted pattern:")
        # print(dataframe.select(polars.col("file_name").str.extract(r"-(.*?)_[^_]+\.html", 1)).head())
        # print("\nSample filename_html values from raw_samples:")
        # print(raw_samples_df.select("filename_html").head())

        dataframe_with_metadata = dataframe.join(
            raw_samples_df,
            left_on=polars.col("file_name").str.extract(r"-(.*?)_[^_]+\.html", 1),
            right_on="filename_html",
            how="left",
        )

        dataframe_with_metadata_filtered = dataframe_with_metadata.drop_nulls(
            subset=["filename_warc", "url", "timestamp", "filename_html"]
        )
        dataframe_with_metadata_filtered = dataframe_with_metadata_filtered.with_columns(
            polars.col("filename_html").str.extract(r"^\d+_(.*)", 1).alias("collection")
        ).drop("filename_html")

        assert dataframe_with_metadata_filtered.height == dataframe.height, (
            "Some rows were lost during metadata restoration!"
        )

        return dataframe_with_metadata_filtered
