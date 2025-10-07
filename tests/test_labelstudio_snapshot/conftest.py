from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class SpanOverlapTestCase:
    test_name: str
    input_intervals: list[dict]
    expected_output: list[dict]


TEST_REMOVE_UNNECESSARY_FIELDS = [
    SpanOverlapTestCase(
        test_name="extra_field",
        input_intervals=[{"globalOffsets": {"start": 0, "end": 5}, "text": "Hello", "extra": "data"}],
        expected_output=[{"globalOffsets": {"start": 0, "end": 5}, "text": "Hello"}],
    ),
    SpanOverlapTestCase(
        test_name="missing_text_field",
        input_intervals=[
            {"globalOffsets": {"start": 0, "end": 5}},  # Missing "text"
            {"globalOffsets": {"start": 10, "end": 15}, "text": "Valid"},
        ],
        expected_output=[{"globalOffsets": {"start": 10, "end": 15}, "text": "Valid"}],
    ),
    SpanOverlapTestCase(
        test_name="normal_case",
        input_intervals=[
            {"globalOffsets": {"start": 0, "end": 5}, "text": "Hello"},
            {"globalOffsets": {"start": 10, "end": 15}, "text": "World"},
        ],
        expected_output=[
            {"globalOffsets": {"start": 0, "end": 5}, "text": "Hello"},
            {"globalOffsets": {"start": 10, "end": 15}, "text": "World"},
        ],
    ),
    SpanOverlapTestCase(
        test_name="extra_field_multiple",
        input_intervals=[
            {"globalOffsets": {"start": 0, "end": 5}, "text": "Hello"},
            {"globalOffsets": {"start": 10, "end": 15}, "text": "World", "extra": "data"},
        ],
        expected_output=[
            {"globalOffsets": {"start": 0, "end": 5}, "text": "Hello"},
            {"globalOffsets": {"start": 10, "end": 15}, "text": "World"},
        ],
    ),
    SpanOverlapTestCase(
        test_name="extra_field_and_missing_text_field",
        input_intervals=[
            {"globalOffsets": {"start": 0, "end": 5}, "text": "Hello"},
            {"globalOffsets": {"start": 10, "end": 15}, "text": "World", "extra": "data"},
            {"globalOffsets": {"start": 20, "end": 25}},  # Missing "text"
        ],
        expected_output=[
            {"globalOffsets": {"start": 0, "end": 5}, "text": "Hello"},
            {"globalOffsets": {"start": 10, "end": 15}, "text": "World"},
        ],
    ),
    SpanOverlapTestCase(
        test_name="missing_globalOffsets_field",
        input_intervals=[
            {"text": "No offset"},  # Missing "globalOffsets"
            {"globalOffsets": {"start": 10, "end": 15}, "text": "Valid"},
        ],
        expected_output=[{"globalOffsets": {"start": 10, "end": 15}, "text": "Valid"}],
    ),
    SpanOverlapTestCase(
        test_name="empty_list",
        input_intervals=[],
        expected_output=[],
    ),
    SpanOverlapTestCase(
        test_name="single_normal_case",
        input_intervals=[{"globalOffsets": {"start": 0, "end": 5}, "text": "Hello"}],
        expected_output=[{"globalOffsets": {"start": 0, "end": 5}, "text": "Hello"}],
    ),
    SpanOverlapTestCase(
        test_name="mixed_fields",
        input_intervals=[
            {"globalOffsets": {"start": 0, "end": 5}, "text": "First", "id": 1},
            {"globalOffsets": {"start": 1, "end": 3}},  # Missing text
            {"globalOffsets": {"start": 10, "end": 15}, "text": "Second", "id": 2},
            {"text": "Third"},  # Missing globalOffsets
        ],
        expected_output=[
            {"globalOffsets": {"start": 0, "end": 5}, "text": "First"},
            {"globalOffsets": {"start": 10, "end": 15}, "text": "Second"},
        ],
    ),
]


TEST_MERGE_OVERLAPPING_TEXT_SPANS = [
    SpanOverlapTestCase(
        test_name="empty_list",
        input_intervals=[],
        expected_output=[],
    ),
    SpanOverlapTestCase(
        test_name="overlapping_intervals_adjacent",
        input_intervals=[
            {"globalOffsets": {"start": 0, "end": 5}, "text": "Hello"},
            {"globalOffsets": {"start": 5, "end": 10}, "text": " World"},
        ],
        expected_output=[{"globalOffsets": {"start": 0, "end": 10}, "text": "Hello World"}],
    ),
    SpanOverlapTestCase(
        test_name="overlapping_intervals_partial",
        input_intervals=[
            {"globalOffsets": {"start": 0, "end": 5}, "text": "Hello"},
            {"globalOffsets": {"start": 3, "end": 7}, "text": "lo Wo"},
        ],
        expected_output=[{"globalOffsets": {"start": 0, "end": 7}, "text": "Hello Wo"}],
    ),
    SpanOverlapTestCase(
        test_name="non_overlapping_intervals",
        input_intervals=[
            {"globalOffsets": {"start": 0, "end": 5}, "text": "Hello"},
            {"globalOffsets": {"start": 6, "end": 10}, "text": "World"},
        ],
        expected_output=[
            {"globalOffsets": {"start": 0, "end": 5}, "text": "Hello"},
            {"globalOffsets": {"start": 6, "end": 10}, "text": "World"},
        ],
    ),
    SpanOverlapTestCase(
        test_name="full_containment_inner",
        input_intervals=[
            {"globalOffsets": {"start": 0, "end": 5}, "text": "Hello"},
            {"globalOffsets": {"start": 2, "end": 5}, "text": "llo"},
        ],
        expected_output=[{"globalOffsets": {"start": 0, "end": 5}, "text": "Hello"}],
    ),
    SpanOverlapTestCase(
        test_name="identical_spans_longer_second",  # not sure if this is possible
        input_intervals=[
            {"globalOffsets": {"start": 0, "end": 5}, "text": "Hello"},
            {"globalOffsets": {"start": 0, "end": 5}, "text": "HelloX"},
        ],
        expected_output=[{"globalOffsets": {"start": 0, "end": 5}, "text": "HelloX"}],
    ),
    SpanOverlapTestCase(
        test_name="identical_spans_longer_first",  # not sure if this is possible
        input_intervals=[
            {"globalOffsets": {"start": 0, "end": 5}, "text": "HelloX"},
            {"globalOffsets": {"start": 0, "end": 5}, "text": "Hello"},
        ],
        expected_output=[{"globalOffsets": {"start": 0, "end": 5}, "text": "HelloX"}],
    ),
    SpanOverlapTestCase(
        test_name="full_containment_outer",
        input_intervals=[
            {"globalOffsets": {"start": 0, "end": 10}, "text": "Hello World"},
            {"globalOffsets": {"start": 2, "end": 5}, "text": "llo"},
        ],
        expected_output=[{"globalOffsets": {"start": 0, "end": 10}, "text": "Hello World"}],
    ),
    SpanOverlapTestCase(
        test_name="touching_intervals",
        input_intervals=[
            {"globalOffsets": {"start": 0, "end": 5}, "text": "Hello"},
            {"globalOffsets": {"start": 5, "end": 10}, "text": " World"},
        ],
        expected_output=[{"globalOffsets": {"start": 0, "end": 10}, "text": "Hello World"}],
    ),
    SpanOverlapTestCase(
        test_name="unsorted_input",
        input_intervals=[
            {"globalOffsets": {"start": 3, "end": 7}, "text": "lo Wo"},
            {"globalOffsets": {"start": 0, "end": 5}, "text": "Hello"},
        ],
        expected_output=[{"globalOffsets": {"start": 0, "end": 7}, "text": "Hello Wo"}],
    ),
    SpanOverlapTestCase(
        test_name="nested_intervals",  # # Nested intervals - outer is larger and should be preferred by sort
        input_intervals=[
            {"globalOffsets": {"start": 0, "end": 5}, "text": "short"},
            {"globalOffsets": {"start": 0, "end": 10}, "text": "long string"},
        ],
        expected_output=[{"globalOffsets": {"start": 0, "end": 10}, "text": "long string"}],
    ),
    SpanOverlapTestCase(
        test_name="multiple_overlaps_complex",  # not sure if this is possible
        input_intervals=[
            {"globalOffsets": {"start": 0, "end": 5}, "text": "Hello"},
            {"globalOffsets": {"start": 3, "end": 7}, "text": "lo Wo"},
            {"globalOffsets": {"start": 6, "end": 10}, "text": "World"},
        ],
        expected_output=[{"globalOffsets": {"start": 0, "end": 10}, "text": "Hello Woorld"}],
    ),
    SpanOverlapTestCase(
        test_name="multiple_overlaps_realistic",  # more realistic case
        input_intervals=[
            {"globalOffsets": {"start": 0, "end": 5}, "text": "Hello"},
            {"globalOffsets": {"start": 3, "end": 8}, "text": "lo Wo"},
            {"globalOffsets": {"start": 6, "end": 10}, "text": "World"},
        ],
        expected_output=[{"globalOffsets": {"start": 0, "end": 10}, "text": "Hello World"}],
    ),
    SpanOverlapTestCase(
        test_name="overlap_with_text_extension",
        input_intervals=[
            {"globalOffsets": {"start": 10, "end": 20}, "text": "This is a "},
            {"globalOffsets": {"start": 15, "end": 25}, "text": "is a test."},
        ],
        expected_output=[{"globalOffsets": {"start": 10, "end": 25}, "text": "This is a test."}],
    ),
    SpanOverlapTestCase(
        test_name="overlap_same_end_longer_text_chosen_from_second_interval",  # not sure if this is possible
        input_intervals=[
            {"globalOffsets": {"start": 0, "end": 10}, "text": "short text"},
            {"globalOffsets": {"start": 5, "end": 10}, "text": "longer example text"},
        ],
        expected_output=[{"globalOffsets": {"start": 0, "end": 10}, "text": "longer example text"}],
    ),
    SpanOverlapTestCase(
        test_name="with_unnecessary_fields",
        input_intervals=[
            {"globalOffsets": {"start": 0, "end": 5}, "text": "Hello", "extra": "field"},
            {"globalOffsets": {"start": 3, "end": 7}, "text": "lo Wo", "another": 123},
        ],
        expected_output=[{"globalOffsets": {"start": 0, "end": 7}, "text": "Hello Wo"}],
    ),
    SpanOverlapTestCase(
        test_name="with_missing_fields",  # should be filtered out
        input_intervals=[
            {"globalOffsets": {"start": 0, "end": 5}},  # Missing text
            {"text": "No offset"},  # Missing globalOffsets
            {"globalOffsets": {"start": 10, "end": 15}, "text": "Valid"},
        ],
        expected_output=[{"globalOffsets": {"start": 10, "end": 15}, "text": "Valid"}],
    ),
    SpanOverlapTestCase(
        test_name="identical_intervals",  # not sure if this is possible
        input_intervals=[
            {"globalOffsets": {"start": 0, "end": 5}, "text": "Hello"},
            {"globalOffsets": {"start": 0, "end": 5}, "text": "Hello"},
        ],
        expected_output=[{"globalOffsets": {"start": 0, "end": 5}, "text": "Hello"}],
    ),
    SpanOverlapTestCase(
        test_name="full_containment_longer_text",  # not sure if this is possible
        input_intervals=[
            {"globalOffsets": {"start": 0, "end": 20}, "text": "This is the full long text"},
            {"globalOffsets": {"start": 5, "end": 15}, "text": "a shorter part"},
        ],
        expected_output=[{"globalOffsets": {"start": 0, "end": 20}, "text": "This is the full long text"}],
    ),
    SpanOverlapTestCase(
        test_name="same_span_longer_text_second",
        # Interval contained, but inner has longer text for the same end span (specific case)
        # This tests the `elif interval["globalOffsets"]["end"] == current_end and len(interval["text"]) > len(merged[-1]["text"])`
        # For this to trigger, the first interval must be shorter or equal in span length
        input_intervals=[
            {"globalOffsets": {"start": 0, "end": 10}, "text": "Short"},
            {
                "globalOffsets": {"start": 0, "end": 10},
                "text": "Longer Text Same Span",
            },  # This interval will update text
        ],
        expected_output=[{"globalOffsets": {"start": 0, "end": 10}, "text": "Longer Text Same Span"}],
    ),
    SpanOverlapTestCase(
        test_name="sorting_same_start_different_end",
        # Test sorting: (start, -end)
        # Interval B starts same as A, but ends later, so B should be processed first.
        input_intervals=[
            {"globalOffsets": {"start": 0, "end": 5}, "text": "Text A"},  # A
            {"globalOffsets": {"start": 0, "end": 10}, "text": "Text B"},  # B
        ],
        # Sorted: B, then A
        # 1. merged = [B]
        # 2. interval A. A.start (0) <= B.end (10). A.end (5) < B.end (10). No change to B.
        expected_output=[{"globalOffsets": {"start": 0, "end": 10}, "text": "Text B"}],
    ),
    SpanOverlapTestCase(
        test_name="complex_sorting_scenario",
        # Test sorting: (start, -end) - more complex
        # C(5,15), D(5,10) -> C then D
        input_intervals=[
            {"globalOffsets": {"start": 0, "end": 5}, "text": "Alpha"},
            {"globalOffsets": {"start": 5, "end": 15}, "text": "Charlie"},  # C
            {"globalOffsets": {"start": 5, "end": 10}, "text": "Delta"},  # D
            {"globalOffsets": {"start": 20, "end": 25}, "text": "Echo"},
        ],
        # Expected sort order for relevant part: Alpha, Charlie, Delta, Echo
        # 1. merged = [Alpha]
        # 2. interval Charlie. start(5) > Alpha.end(5) is false. Merge.
        #    current_end = 5. Charlie.end(15) > 5. additional_text = Charlie.text.
        #    merged[-1] = {0,15, "AlphaCharlie"}
        # 3. interval Delta. start(5) > merged.end(15) is false. Merge.
        #    current_end = 15. Delta.end(10) < 15. No change.
        # 4. merged = [Echo]
        expected_output=[
            {"globalOffsets": {"start": 0, "end": 15}, "text": "AlphaCharlie"},
            {"globalOffsets": {"start": 20, "end": 25}, "text": "Echo"},
        ],
    ),
]
