from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class CalculateGenericTestCase:
    test_name: str
    reference: str | list[str]
    candidate: str | list[str]


TEST_VALID_INPUT_STRING = [
    CalculateGenericTestCase(reference="hello world", candidate="hello", test_name="partial_ratio"),
    CalculateGenericTestCase(reference="hello world", candidate="world", test_name="partial_ratio"),
    CalculateGenericTestCase(reference="hello world", candidate="hello world", test_name="same"),
    CalculateGenericTestCase(reference="hello world", candidate="", test_name="empty_candidate"),
    CalculateGenericTestCase(reference="", candidate="hello world", test_name="empty_reference"),
    CalculateGenericTestCase(reference="", candidate="", test_name="both_empty"),
    CalculateGenericTestCase(
        reference="tide",
        candidate="diet",
        test_name="partial_ratio_anagram_0",
    ),
    CalculateGenericTestCase(reference="diet", candidate="tide", test_name="partial_ratio_anagram_1"),
    CalculateGenericTestCase(
        reference="conversation",
        candidate="voices rant on",
        test_name="partial_ratio_anagram_2",
    ),
    CalculateGenericTestCase(
        reference="the eyes",
        candidate="they see",
        test_name="partial_ratio_anagram_3",
    ),
]
TEST_VALID_INPUT_TOKEN = [
    CalculateGenericTestCase(reference=["hello", "world"], candidate=["hello"], test_name="list_partial_match_1"),
    CalculateGenericTestCase(reference=["hello", "world"], candidate=["world"], test_name="list_partial_match_2"),
    CalculateGenericTestCase(reference=["hello", "world"], candidate=["hello", "world"], test_name="list_same"),
    CalculateGenericTestCase(reference=["hello", "world"], candidate=[], test_name="list_empty_candidate"),
    CalculateGenericTestCase(reference=[], candidate=["hello", "world"], test_name="list_empty_reference"),
    CalculateGenericTestCase(reference=[], candidate=[], test_name="list_both_empty"),
    CalculateGenericTestCase(
        reference=["the", "cat", "sat"], candidate=["the", "cat"], test_name="list_partial_match_3"
    ),
    CalculateGenericTestCase(
        reference=["e", "x", "a", "m", "p", "l", "e"], candidate=["e", "x", "a", "m"], test_name="list_partial_match_4"
    ),
    CalculateGenericTestCase(
        reference=["a", "b", "c", "d", "e", "f"], candidate=["a", "b", "c", "d"], test_name="list_partial_match_5"
    ),
    CalculateGenericTestCase(
        reference=["t", "i", "d", "e"], candidate=["d", "i", "t", "e"], test_name="list_anagram_0"
    ),
    CalculateGenericTestCase(
        reference=["d", "i", "e", "t"], candidate=["t", "i", "d", "e"], test_name="list_anagram_1"
    ),
    CalculateGenericTestCase(
        reference=["c", "o", "n", "v", "e", "r", "s", "a", "t", "i", "o", "n"],
        candidate=["v", "o", "i", "c", "e", "s", "r", "a", "n", "t", "o", "n"],
        test_name="list_anagram_2",
    ),
    CalculateGenericTestCase(
        reference=["t", "h", "e", "e", "y", "e", "s"],
        candidate=["t", "h", "e", "y", "s", "e", "e"],
        test_name="list_anagram_3",
    ),
]


TEST_VALID_INPUT_SAME_STRING = [
    CalculateGenericTestCase(reference="hello world", candidate="hello world", test_name="same"),
    CalculateGenericTestCase(reference="", candidate="", test_name="both_empty"),
]

TEST_VALID_INPUT_SAME_TOKEN = [
    CalculateGenericTestCase(reference=["hello", "world"], candidate=["hello", "world"], test_name="list_same"),
    CalculateGenericTestCase(reference=[], candidate=[], test_name="list_both_empty"),
]


@dataclass(slots=True, frozen=True)
class RapidFuzzInvalidInputTestCase:
    test_name: str
    reference: Any
    candidate: Any
    expected_exception: type[Exception]


TEST_INVALID_INPUT_RAPIDFUZZ = [
    RapidFuzzInvalidInputTestCase(
        reference=None, candidate="hello", expected_exception=TypeError, test_name="none_reference"
    ),
    RapidFuzzInvalidInputTestCase(
        reference="hello", candidate=None, expected_exception=TypeError, test_name="none_candidate"
    ),
    RapidFuzzInvalidInputTestCase(reference=None, candidate=None, expected_exception=TypeError, test_name="none_both"),
    RapidFuzzInvalidInputTestCase(
        reference=123, candidate="hello", expected_exception=TypeError, test_name="invalid_type_reference"
    ),
    RapidFuzzInvalidInputTestCase(
        reference="hello", candidate=456, expected_exception=TypeError, test_name="invalid_type_candidate"
    ),
    RapidFuzzInvalidInputTestCase(
        reference=123, candidate=456, expected_exception=TypeError, test_name="invalid_type_both"
    ),
]


@dataclass(slots=True, frozen=True)
class CyDiffLibInvalidInputTestCase:
    test_name: str
    reference: Any
    candidate: Any
    expected_exception: type[Exception] | tuple[type[Exception], ...]


TEST_INVALID_INPUT_CYDIFFLIB = [
    CyDiffLibInvalidInputTestCase(
        reference=None, candidate="hello", expected_exception=(TypeError, AttributeError), test_name="none_reference"
    ),
    CyDiffLibInvalidInputTestCase(
        reference="hello", candidate=None, expected_exception=(TypeError, AttributeError), test_name="none_candidate"
    ),
    CyDiffLibInvalidInputTestCase(
        reference=None, candidate=None, expected_exception=(TypeError, AttributeError), test_name="none_both"
    ),
    CyDiffLibInvalidInputTestCase(
        reference=123, candidate="hello", expected_exception=TypeError, test_name="invalid_type_reference"
    ),
    CyDiffLibInvalidInputTestCase(
        reference="hello", candidate=456, expected_exception=TypeError, test_name="invalid_type_candidate"
    ),
    CyDiffLibInvalidInputTestCase(
        reference=123, candidate=456, expected_exception=TypeError, test_name="invalid_type_both"
    ),
]


@dataclass(slots=True, frozen=True)
class CustomTokenInvalidInputTestCase:
    test_name: str
    reference: Any
    candidate: Any
    expected_exception: type[Exception]


TEST_INVALID_INPUT_CUSTOM_TOKEN = [
    CustomTokenInvalidInputTestCase(
        reference=None, candidate=["hello"], expected_exception=TypeError, test_name="none_reference"
    ),
    CustomTokenInvalidInputTestCase(
        reference=["hello"], candidate=None, expected_exception=TypeError, test_name="none_candidate"
    ),
    CustomTokenInvalidInputTestCase(
        reference=None, candidate=None, expected_exception=TypeError, test_name="none_both"
    ),
    CustomTokenInvalidInputTestCase(
        reference="hello", candidate=["hello"], expected_exception=TypeError, test_name="invalid_type_reference"
    ),
    CustomTokenInvalidInputTestCase(
        reference=["hello"], candidate="hello", expected_exception=TypeError, test_name="invalid_type_candidate"
    ),
    CustomTokenInvalidInputTestCase(
        reference="hello", candidate="hello", expected_exception=TypeError, test_name="invalid_type_both"
    ),
]
