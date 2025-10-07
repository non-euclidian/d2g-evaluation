from datetime import datetime

import pydantic


@pydantic.dataclasses.dataclass
class CompletedBy:
    id: int
    email: str
    first_name: str
    last_name: str


@pydantic.dataclasses.dataclass
class Result:
    id: str
    type: str
    value: dict
    origin: str
    to_name: str
    from_name: str


@pydantic.dataclasses.dataclass
class Annotations:
    id: int
    completed_by: CompletedBy
    result: list[Result]
    reviews: list
    was_cancelled: bool
    ground_truth: bool
    created_at: str
    updated_at: str
    draft_created_at: datetime | None
    lead_time: float
    prediction: dict
    result_count: int
    unique_id: str
    import_id: None
    last_action: str
    task: int
    project: int
    updated_by: int
    parent_prediction: None
    parent_annotation: None
    last_created_by: int

    def __post_init__(self) -> None:
        if isinstance(self.completed_by, dict):
            self.completed_by = CompletedBy(**self.completed_by)
        if isinstance(self.result, list):
            self.result = [Result(**result) if isinstance(result, dict) else result for result in self.result]


@pydantic.dataclasses.dataclass
class LabelStudioTask:
    id: int
    annotations: list[Annotations]
    file_upload: str
    drafts: list
    predictions: list
    agreement: float | None
    data: dict
    meta: dict
    created_at: str
    updated_at: str
    inner_id: int
    total_annotations: int
    cancelled_annotations: int
    total_predictions: int
    comment_count: int
    unresolved_comment_count: int
    last_comment_updated_at: str | None
    project: int
    updated_by: int
    comment_authors: list

    def __post_init__(self) -> None:
        self.annotations = [
            Annotations(**annotation) if isinstance(annotation, dict) else annotation for annotation in self.annotations
        ]


@pydantic.dataclasses.dataclass
class ProcessedData:
    task_id: int
    file_name: str
    annotation_id: int
    annotated_by: str
    check_loading: int | None
    check_not_empty: int | None
    check_lang_match: int | None
    check_harmful_content: int | None
    check_highlightable: int | None
    rate_conf_yourself: int | None
    original_spans: list[dict]
    len_original_spans: int
    clean_spans: list[dict]
    len_clean_spans: int
    html: str
    lead_time: float
    raw_string: str
    len_raw_string: int
    spans: list[str]
    len_spans: int
