import pathlib
from typing import Any, ClassVar

import datasets  # type: ignore

from d2g_evaluation.evaluation.base_evaluation import (
    BaseEvaluation,
    PairwiseEvaluationResult,
    SampleEvaluationResult,
)


class HumanVsToolEvaluation(BaseEvaluation):
    """Evaluate tool outputs against human annotations."""

    EVALUATION_TYPE: ClassVar[str] = "eval_human_vs_tool"
    INSUFFICIENT_ANNOTATIONS: ClassVar[int] = 1  # for human vs tool it's possible to have 1 annotation

    # dataset output columns
    SAMPLE_RESULT_COLUMN: ClassVar[str] = "sample_result_human_vs_tool"
    PAIRWISE_RESULTS_COLUMN: ClassVar[str] = "pairwise_results_human_vs_tool"

    # dataset specific columns and fields
    _TASK_ID_COLUMN: ClassVar[str] = "task_id"
    _ANNOTATIONS_COLUMN: ClassVar[str] = "annotations"

    _FILTER_CRITERION_FIELD: ClassVar[str] = "len_merged_spans"
    _ANNOTATOR_ID_FIELD: ClassVar[str] = "annotated_by"

    def __init__(self) -> None:
        super().__init__()

    def _process_sample(
        self,
        sample: dict,
        tool_column: str,
        tool_name: str,
        text_field: str = "as_string",
        limit_annotators: int | None = None,
        **calculate_kwargs: Any,  # noqa: ANN401
    ) -> dict[str, dict | list[dict]]:
        task_id = sample[self._TASK_ID_COLUMN]
        metric_name = calculate_kwargs["metric_name"]

        task_id = sample[self._TASK_ID_COLUMN]
        metric_name = calculate_kwargs["metric_name"]

        self.logger.debug(
            "Processing sample (task_id=%s) with tool '%s' using metric '%s'", task_id, tool_name, metric_name
        )
        self.logger.debug("Passed calculate_kwargs: %s", calculate_kwargs)

        filtered_annotations = self._filter_annotations_with_no_text(annotations=sample[self._ANNOTATIONS_COLUMN])

        self.logger.debug(
            "Handle insufficient annotations (if less than %d annotations)...", self.INSUFFICIENT_ANNOTATIONS
        )
        if len(filtered_annotations) < self.INSUFFICIENT_ANNOTATIONS:
            sample_result, pairwise_results = self._handle_insufficient_annotations(
                task_id=task_id,
                num_annotations=len(filtered_annotations),
                metric_name=metric_name,
            )
            return self._build_result_dict(sample_result=sample_result, pairwise_results=pairwise_results)

        if limit_annotators is not None and len(filtered_annotations) > limit_annotators:
            self.logger.debug(
                "Limiting number of annotators to %d (from %d)", limit_annotators, len(filtered_annotations)
            )
            filtered_annotations = self.sample_with_limit(items=filtered_annotations, limit=limit_annotators)
            self.logger.debug("Number of annotations after limiting: %d", len(filtered_annotations))

        prep_pairs = self.generate_one_to_many_pairs(
            main_item=sample[tool_column],
            items=filtered_annotations,
        )

        pairwise_results = []

        for tool_output, annotation in prep_pairs:
            pair_result = self.interface_metrics.calculate(
                reference=annotation[text_field],
                candidate=tool_output,
                **calculate_kwargs,
            )
            pairwise_results.append(
                PairwiseEvaluationResult(
                    metric_computation=pair_result,
                    task_id=task_id,
                    reference_id=annotation[self._ANNOTATOR_ID_FIELD],
                    candidate_id=tool_name,
                )
            )

        sample_result = SampleEvaluationResult.from_pairwise_results(
            metric_name=metric_name,
            task_id=task_id,
            pairwise_results=pairwise_results,
        )

        return self._build_result_dict(sample_result=sample_result, pairwise_results=pairwise_results)

    def evaluate(  # noqa: PLR0913
        self,
        dataset: datasets.Dataset | str | pathlib.Path,
        *,
        tool_column: str,
        tool_name: str,
        text_field: str = "as_string",
        limit_annotators: int | None = None,
        datasets_map_params: dict[str, Any] | None = None,
        **calculate_kwargs: Any,  # noqa: ANN401
    ) -> datasets.Dataset:
        self.logger.debug("Starting evaluation with kwargs: %s", calculate_kwargs)

        if "metric_name" not in calculate_kwargs:
            msg = "The 'metric_name' argument must be provided as a keyword argument."
            raise ValueError(msg)

        dataset = (
            self.load_hf_dataset_from_file(file_path=dataset) if isinstance(dataset, (str, pathlib.Path)) else dataset
        )

        fn_kwargs = {
            "tool_column": tool_column,
            "tool_name": tool_name,
            "text_field": text_field,
            "limit_annotators": limit_annotators,
            **calculate_kwargs,
        }

        default_map_params = {
            "fn_kwargs": fn_kwargs,
            "load_from_cache_file": False,
            # "remove_columns": dataset.column_names,
            "desc": f"Evaluating with metric '{calculate_kwargs['metric_name']}'",
        }

        if datasets_map_params:
            default_map_params.update(datasets_map_params)

        self.logger.debug("Calling dataset.map() with parameters: %s", default_map_params)

        ds = dataset.map(self._process_sample, **default_map_params)
        return ds  # noqa: RET504
