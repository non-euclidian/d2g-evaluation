import logging
from typing import ClassVar


class SpanOverlapMerger:
    NECESSAR_FIELDS: ClassVar[set[str]] = {"globalOffsets", "text"}

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def _remove_unnecessary_fields(self, intervals: list[dict] | list) -> list[dict]:
        cleaned_intervals = []
        error_count = 0

        for interval in intervals:
            # validate keys before attempting to access them
            if all(field in interval for field in self.NECESSAR_FIELDS):
                cleaned_interval = {key: interval[key] for key in self.NECESSAR_FIELDS}
                cleaned_intervals.append(cleaned_interval)
            else:
                error_count += 1
                missing_fields = [field for field in self.NECESSAR_FIELDS if field not in interval]
                self.logger.error(
                    "Missing required fields: %s. Interval: %s. Available keys: %s",
                    missing_fields,
                    interval,
                    interval.keys(),
                )
                self.logger.error("CHECK THIS - IT'S UNKNOWN BEHAVIOUR FROM THE LABEL STUDIO")
                self.logger.error("Count this as an error: %d", error_count)
                # continue processing the next interval
                continue

        return cleaned_intervals

    def merge_overlapping_text_spans(self, intervals: list[dict] | list) -> list[dict]:
        self.logger.debug("Input intervals: %d", len(intervals))
        if not intervals:
            return []

        # remove unnecessary fields
        intervals = self._remove_unnecessary_fields(intervals)

        # sort the intervals based on the starting points, then by end points (for nested intervals)
        sorted_intervals = sorted(intervals, key=lambda x: (x["globalOffsets"]["start"], -x["globalOffsets"]["end"]))

        merged: list[dict] = []
        for interval in sorted_intervals:
            if not merged or interval["globalOffsets"]["start"] > merged[-1]["globalOffsets"]["end"]:
                # if there's no overlap, add the interval as is
                merged.append(interval.copy())
            else:
                # if there's an overlap, merge the intervals
                current_end = merged[-1]["globalOffsets"]["end"]
                if interval["globalOffsets"]["end"] > current_end:
                    # extend the existing interval
                    additional_text = interval["text"][current_end - interval["globalOffsets"]["start"] :]
                    merged[-1]["text"] += additional_text
                    merged[-1]["globalOffsets"]["end"] = interval["globalOffsets"]["end"]
                elif interval["globalOffsets"]["end"] == current_end and len(interval["text"]) > len(
                    merged[-1]["text"]
                ):
                    # if the new interval has the same end but more text, use it
                    merged[-1]["text"] = interval["text"]

        self.logger.debug("Merged intervals (ordered): %d", len(merged))

        assert len(merged) <= len(intervals)
        assert all(merged[i]["globalOffsets"]["start"] <= merged[i]["globalOffsets"]["end"] for i in range(len(merged)))
        assert all(
            merged[i]["globalOffsets"]["end"] <= merged[i + 1]["globalOffsets"]["start"] for i in range(len(merged) - 1)
        )
        return merged
