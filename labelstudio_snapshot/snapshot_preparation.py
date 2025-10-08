from __future__ import annotations

import json
import logging
import pathlib

import polars  # noqa: ICN001
import tqdm  # type: ignore

from labelstudio_snapshot.merge_overlap import SpanOverlapMerger
from labelstudio_snapshot.snapshot_dataclasses import Annotations, LabelStudioTask, ProcessedData, Result


class SnapshotPreparationPipeline:
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.overlap_span_merger = SpanOverlapMerger()

    def read_json(self, json_path: str) -> list[dict]:
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

            return ProcessedData(
                task_id=task.id,
                file_name=task.file_upload,
                annotation_id=annotation.id,
                annotated_by=annotation.completed_by.email,
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
                clean_spans=merged_overlapped_spans,
                len_clean_spans=len(merged_overlapped_spans),
                html=task.data["html"],
                lead_time=annotation.lead_time,
                raw_string=raw_string,  # our main string
                len_raw_string=len(raw_string),
                spans=spans,
                len_spans=len(spans),
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
        self, json_path: str
    ) -> tuple[list[dict], list[LabelStudioTask], list[ProcessedData], polars.DataFrame]:
        json_data = self.read_json(json_path)
        structured_data = self.create_structured_data(json_data)
        processed_structured_data = self.process_structured_data(structured_data)
        dataframe = self.convert_processed_data_to_dataframe(processed_structured_data)

        return json_data, structured_data, processed_structured_data, dataframe


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
