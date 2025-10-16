import pathlib
from typing import Any, ClassVar

import datasets  # type: ignore

from d2g_evaluation.evaluation.base_evaluation import (
    BaseEvaluation,
    PairwiseEvaluationResult,
    SampleEvaluationResult,
)
from d2g_evaluation.metrics.metrics_core import (
    ImplementedCustomMetrics,
    ImplementedCyDiffLibMetrics,
    ImplementedRapidFuzzMetrics,
)


class HumanVsHumanEvaluation(BaseEvaluation):
    PIPELINE_NAME: ClassVar[str] = "eval_human_vs_human"
    INSUFFICIENT_ANNOTATIONS: ClassVar[int] = 2

    # dataset output columns
    SAMPLE_RESULT_COLUMN: ClassVar[str] = "sample_result"
    PAIRWISE_RESULTS_COLUMN: ClassVar[str] = "pairwise_results"

    # dataset speceific columns and fields
    _TASK_ID_COLUMN: ClassVar[str] = "task_id"
    _ANNOTATIONS_COLUMN: ClassVar[str] = "annotations"

    _FILTER_CRITERION_FIELD: ClassVar[str] = "len_merged_spans"
    _ANNOTATOR_ID_FIELD: ClassVar[str] = "annotated_by"

    def __init__(self) -> None:
        super().__init__()

    def _filter_annotations_with_no_text(self, annotations: list[dict]) -> list[dict]:
        """Remove annotations with no text (len_merged_spans == 0)."""
        self.logger.debug("Before filtering, number of annotations: %d", len(annotations))
        filtered_annotations = [ann for ann in annotations if ann[self._FILTER_CRITERION_FIELD] != 0]
        self.logger.debug("After filtering, number of annotations: %d", len(filtered_annotations))
        return filtered_annotations

    def _handle_insufficient_annotations(
        self,
        task_id: int | str | None,
        num_annotations: int,
        metric_name: ImplementedRapidFuzzMetrics | ImplementedCyDiffLibMetrics | ImplementedCustomMetrics | str,
    ) -> tuple[SampleEvaluationResult, list[PairwiseEvaluationResult]]:
        """Handle samples with insufficient valid annotations."""
        self.logger.warning(
            "Sample with (task_id: %s) has less than %d valid annotations after filtering. Skipping.",
            task_id,
            self.INSUFFICIENT_ANNOTATIONS,
        )
        sample_result = SampleEvaluationResult.from_pairwise_results(metric_name=metric_name, task_id=task_id)
        pairwise_results = [
            PairwiseEvaluationResult(
                metric_computation=None,
                task_id=task_id,
                reference_id=None,
                candidate_id=None,
            )
            for _ in range(num_annotations)
        ]
        return sample_result, pairwise_results

    def _build_result_dict(
        self, sample_result: SampleEvaluationResult, pairwise_results: list[PairwiseEvaluationResult]
    ) -> dict[str, dict | list[dict]]:
        """Convert evaluation results to dictionary format."""
        return {
            "sample_result": sample_result.to_dict(),
            "pairwise_results": [pr.to_dict() for pr in pairwise_results],
        }

    def _process_sample(
        self,
        sample: dict,
        text_field: str = "as_string",
        limit_annotators: int | None = None,
        **calculate_kwargs: Any,  # noqa: ANN401
    ) -> dict[str, dict | list[dict]]:
        task_id = sample[self._TASK_ID_COLUMN]
        metric_name = calculate_kwargs["metric_name"]
        self.logger.debug("Processing sample (task_id: %s) for metric '%s'", task_id, metric_name)
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

        self.logger.debug("Compute pairwise comparisons...")
        prep_pairs = self.generate_pairwise_combinations(items=filtered_annotations)
        self.logger.debug("Number of pairwise combinations: %d", len(prep_pairs))

        pairwise_results = []
        for ref_ann, cand_ann in prep_pairs:
            pair_result = self.interface_metrics.calculate(
                reference=ref_ann[text_field],
                candidate=cand_ann[text_field],
                **calculate_kwargs,
            )
            pairwise_results.append(
                PairwiseEvaluationResult(
                    metric_computation=pair_result,
                    task_id=task_id,
                    reference_id=ref_ann[self._ANNOTATOR_ID_FIELD],
                    candidate_id=cand_ann[self._ANNOTATOR_ID_FIELD],
                )
            )
        sample_result = SampleEvaluationResult.from_pairwise_results(
            metric_name=metric_name,
            task_id=task_id,
            pairwise_results=pairwise_results,
        )
        return self._build_result_dict(sample_result=sample_result, pairwise_results=pairwise_results)

    def evaluate(
        self,
        dataset: datasets.Dataset | str | pathlib.Path,
        *,
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
