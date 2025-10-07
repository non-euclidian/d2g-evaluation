import pytest

from labelstudio_snapshot.merge_overlap import SpanOverlapMerger
from tests.test_labelstudio_snapshot.conftest import (
    TEST_MERGE_OVERLAPPING_TEXT_SPANS,
    TEST_REMOVE_UNNECESSARY_FIELDS,
    SpanOverlapTestCase,
)


class TestSpanOverlapMerger:
    @pytest.fixture
    def merger(self) -> SpanOverlapMerger:
        return SpanOverlapMerger()

    @pytest.mark.parametrize("test_case", TEST_REMOVE_UNNECESSARY_FIELDS)
    def test_remove_unnecessary_fields(
        self,
        merger: SpanOverlapMerger,
        test_case: SpanOverlapTestCase,
    ) -> None:
        result = merger._remove_unnecessary_fields(test_case.input_intervals)
        assert result == test_case.expected_output

    @pytest.mark.parametrize("test_case", TEST_MERGE_OVERLAPPING_TEXT_SPANS)
    def test_merge_overlapping_text_spans(self, merger: SpanOverlapMerger, test_case: SpanOverlapTestCase) -> None:
        result = merger.merge_overlapping_text_spans(test_case.input_intervals)
        assert result == test_case.expected_output
