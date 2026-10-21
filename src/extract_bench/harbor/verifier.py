"""Harbor's file boundary into the existing ExtractBench evaluator."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from extract_bench.evaluation.evaluators.extract import ExtractEvaluator
from extract_bench.evaluation.metrics.extract.unified_evidence_metric import ValueCell
from extract_bench.evaluation.score_case import load_extract_test_case
from extract_bench.schemas.extract_output import ExtractOutput
from extract_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from extract_bench.schemas.product import ProductType


def _feedback(cells: Sequence[ValueCell], metadata: Mapping[str, Any]) -> dict[str, Any]:
    expected = sum(cell.gt_path is not None for cell in cells)
    predicted = sum(cell.pred_path is not None for cell in cells)
    matched = sum(cell.value_correct for cell in cells)
    if (expected, predicted, matched) != (
        metadata["expected_cells"],
        metadata["predicted_cells"],
        metadata["v_correct"],
    ):
        raise ValueError("Value feedback does not account for the scorer's cell counts")
    denominator = expected + predicted
    entries = []
    for cell in cells:
        if cell.gt_path is None:
            outcome = "prediction_only"
        elif cell.pred_path is None:
            outcome = "reference_only"
        else:
            outcome = "matched" if cell.value_correct else "value_differs"
        entries.append(
            {
                "reference_path": cell.gt_path,
                "output_path": cell.pred_path,
                "outcome": outcome,
                "f1_credit": 2.0 / denominator if cell.value_correct else 0.0,
            }
        )
    return {
        "schema_version": 1,
        "status": "scored",
        "path_space": "normalized_scoring_inputs",
        "counts": {"expected": expected, "predicted": predicted, "matched": matched},
        "value_incomplete": metadata["value_incomplete"],
        "cells": entries,
    }


def score_prediction(test_json: Path, prediction: Any) -> tuple[dict[str, float], dict[str, Any]]:
    """Use exactly the normal evaluator's normalization and metric selection."""
    case = load_extract_test_case(test_json, source_file=Path("/app/document.pdf"))
    now = datetime.now(UTC)
    result = InferenceResult(
        request=InferenceRequest(
            example_id=case.test_id,
            source_file_path=str(case.file_path),
            product_type=ProductType.EXTRACT,
        ),
        pipeline_name="harbor",
        product_type=ProductType.EXTRACT,
        raw_output={},
        output=ExtractOutput(
            example_id=case.test_id,
            pipeline_name="harbor",
            extracted_data=prediction,
            field_citations=[],
        ),
        started_at=now,
        completed_at=now,
        latency_in_ms=0,
    )
    cells: list[ValueCell] = []
    bundle = ExtractEvaluator().compute_metrics(result, case, value_cells=cells)
    metrics = {
        metric.metric_name: metric.value
        for metric in bundle.metrics
        if metric.metric_name.startswith("extract_unified_value_")
    }
    value_f1 = next(metric for metric in bundle.metrics if metric.metric_name == "extract_unified_value_f1")
    return metrics, _feedback(cells, value_f1.metadata)


def verify(test_json: Path, prediction_path: Path, log_dir: Path) -> dict[str, Any]:
    """Bad/missing submissions earn zero; broken grading raises an error.

    Only submission decoding is caught. A missing reference, loader error or
    scorer failure must fail the verifier instead of becoming a model score.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    reward_path = log_dir / "reward.txt"
    for name in ("reward.txt", "metrics.json", "feedback.json"):
        (log_dir / name).unlink(missing_ok=True)
    try:
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as error:
        report: dict[str, Any] = {"status": "invalid_submission", "error": str(error), "value_f1": 0.0}
    else:
        if not isinstance(prediction, dict):
            report = {"status": "invalid_submission", "error": "Expected one JSON object", "value_f1": 0.0}
        else:
            metrics, feedback = score_prediction(test_json, prediction)
            report = {
                "status": "scored",
                "value_f1": metrics["extract_unified_value_f1"],
                "metrics": metrics,
            }
    if report["status"] == "invalid_submission":
        feedback = {"schema_version": 1, "status": "invalid_submission", "error": report["error"], "cells": []}
    report["perfect"] = int(report["value_f1"] == 1.0)
    (log_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (log_dir / "feedback.json").write_text(json.dumps(feedback, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    reward_path.write_text(f"{report['value_f1']}\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-json", type=Path, default=Path("/tests/reference.test.json"))
    parser.add_argument("--prediction", type=Path, default=Path("/app/output.json"))
    parser.add_argument("--log-dir", type=Path, default=Path("/logs/verifier"))
    args = parser.parse_args()
    verify(args.test_json, args.prediction, args.log_dir)


if __name__ == "__main__":
    main()
