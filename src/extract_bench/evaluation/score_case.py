"""Score one extraction prediction against one test case.

The evaluation runner scores a whole results directory. Interactive harnesses
that produce a single prediction — the Extract tool-sandbox dogfood harness is
the current caller — need the same metric for one document, without inventing
their own rule parsing or their own copy of the metric call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from extract_bench.evaluation.metrics.extract import (
    compute_unified_evidence_metrics,
)
from extract_bench.test_cases.loader import load_test_case
from extract_bench.test_cases.schema import ExtractTestCase

SCORER_NAME = "extract_bench.extract_unified_value_f1"
_VALUE_METRIC_PREFIX = "extract_unified_value_"
_VALUE_F1_METRIC = "extract_unified_value_f1"


def load_extract_test_case(test_json: Path, source_file: Path | None = None) -> ExtractTestCase:
    """Load an extract test case from its ``.test.json``.

    ``source_file`` is the document the case is about. The loader derives it
    from the sibling naming convention when it is not given, which is the
    common case for a single-document dataset directory.
    """
    resolved = source_file or _sibling_source(test_json)
    case = load_test_case(resolved, test_json)
    if case is None:
        raise ValueError(f"No test case loaded from {test_json}")
    if not isinstance(case, ExtractTestCase):
        raise ValueError(f"{test_json} is a {type(case).__name__}, not an extract test case")
    return case


def _sibling_source(test_json: Path) -> Path:
    """Find the document a ``<stem>.test.json`` describes, by stem match."""
    stem = test_json.name.removesuffix(".test.json")
    candidates = [
        path
        for path in sorted(test_json.parent.glob(f"{stem}.*"))
        if path.suffix not in (".json", ".log") and path.is_file()
    ]
    if not candidates:
        raise ValueError(f"No source document beside {test_json}; pass source_file explicitly")
    return candidates[0]


def score_extract_prediction(case: ExtractTestCase, prediction: Any) -> dict[str, Any]:
    """Value-F1 metrics for one prediction, or an unscorable verdict.

    A case with neither expected output nor field rules has nothing to compare
    against; that is reported rather than scored as zero, so a caller can tell
    "the harness could not grade this" from "the extraction was wrong".
    """
    rules = case.get_extract_field_rules()
    expected = case.expected_output
    if not expected and not rules:
        return {
            "scorer": SCORER_NAME,
            "scorable": False,
            "reason": "The test case has no expected output or field rules.",
            "metrics": {},
        }
    metrics = compute_unified_evidence_metrics(
        expected or {},
        prediction,
        rules,
        data_schema=case.data_schema,
    )
    by_name = {metric.metric_name: metric for metric in metrics}
    return {
        "scorer": SCORER_NAME,
        "scorable": True,
        "metrics": {name: metric.value for name, metric in by_name.items() if name.startswith(_VALUE_METRIC_PREFIX)},
        "value_f1_metadata": (by_name[_VALUE_F1_METRIC].metadata if _VALUE_F1_METRIC in by_name else None),
    }
