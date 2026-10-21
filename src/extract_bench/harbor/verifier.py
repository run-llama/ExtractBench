"""Harbor's file boundary into the existing ExtractBench evaluator."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from extract_bench.evaluation.evaluators.extract import ExtractEvaluator
from extract_bench.evaluation.score_case import load_extract_test_case
from extract_bench.schemas.extract_output import ExtractOutput
from extract_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from extract_bench.schemas.product import ProductType


def score_prediction(test_json: Path, prediction: Any) -> dict[str, float]:
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
    bundle = ExtractEvaluator().compute_metrics(result, case)
    return {
        metric.metric_name: metric.value
        for metric in bundle.metrics
        if metric.metric_name.startswith("extract_unified_value_")
    }


def verify(test_json: Path, prediction_path: Path, log_dir: Path) -> dict[str, Any]:
    """Bad/missing submissions earn zero; broken grading raises an error.

    Only submission decoding is caught. A missing reference, loader error or
    scorer failure must fail the verifier instead of becoming a model score.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    reward_path = log_dir / "reward.txt"
    reward_path.unlink(missing_ok=True)
    try:
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as error:
        report: dict[str, Any] = {"status": "invalid_submission", "error": str(error), "value_f1": 0.0}
    else:
        if not isinstance(prediction, dict):
            report = {"status": "invalid_submission", "error": "Expected one JSON object", "value_f1": 0.0}
        else:
            metrics = score_prediction(test_json, prediction)
            report = {
                "status": "scored",
                "value_f1": metrics["extract_unified_value_f1"],
                "metrics": metrics,
            }
    report["perfect"] = int(report["value_f1"] == 1.0)
    (log_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
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
