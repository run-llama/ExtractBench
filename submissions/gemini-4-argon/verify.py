"""Replay this submission from saved predictions and a pinned public dataset; no inference."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from extract_bench.evaluation.metrics.extract.unified_evidence_metric import compute_unified_evidence_metrics
from extract_bench.test_cases.schema import ExtractFieldTestRule

SUBMISSION = Path(__file__).resolve().parent
REPO = SUBMISSION.parents[1]
METRICS = {
    "value_f1": "extract_unified_value_f1",
    "word_grounding_f1": "extract_unified_grounded_f1",
    "page_grounding_f1": "extract_unified_page_f1",
}


def aggregate(rows: Sequence[Mapping[str, str]]) -> dict[str, str]:
    grounded = [row for row in rows if row["grounding_eligible"] == "True"]
    return {
        "Documents": str(len(rows)),
        "Grounding_Documents": str(len(grounded)),
        "Perfect_Documents": f"{100 * sum(float(row['value_f1']) == 1 for row in rows) / len(rows):.2f}",
        "Value_F1": f"{100 * sum(float(row['value_f1']) for row in rows) / len(rows):.2f}",
        "Word_Grounding": f"{100 * sum(float(row['word_grounding_f1']) for row in grounded) / len(grounded):.2f}",
        "Page_Grounding": f"{100 * sum(float(row['page_grounding_f1']) for row in grounded) / len(grounded):.2f}",
        "Cost_Per_Page": f"{sum(float(row['cost_usd']) / int(row['pages']) for row in rows) / len(rows):.6f}",
    }


def verify(snapshot: Path) -> None:
    config = json.loads((SUBMISSION / "config.json").read_text())
    with (SUBMISSION / "documents.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    with gzip.open(SUBMISSION / "predictions.jsonl.gz", "rt") as stream:
        prediction_rows = [json.loads(line) for line in stream]
    predictions = {row["id"]: row for row in prediction_rows}
    assert len(predictions) == len(prediction_rows), "Duplicate prediction IDs"
    dataset = {}
    for cohort, expected_hash in config["dataset"]["split_sha256"].items():
        content = (snapshot / f"{cohort}.jsonl").read_bytes()
        assert hashlib.sha256(content).hexdigest() == expected_hash, f"Dataset revision mismatch: {cohort}"
        for line in content.splitlines():
            case = json.loads(line)
            assert case["id"] not in dataset, f"Duplicate dataset ID: {case['id']}"
            dataset[case["id"]] = case
    assert len(rows) == len(dataset) == len({row["id"] for row in rows})
    assert {row["id"] for row in rows} == set(dataset), "Document coverage mismatch"
    expected_predictions = set()
    for row in rows:
        case = dataset[row["id"]]
        schema = json.loads(case["data_schema"])
        canonical_schema = json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        assert hashlib.sha256(canonical_schema).hexdigest() == row["schema_sha256"], row["id"]
        rules = [
            ExtractFieldTestRule.model_validate({**rule, "field_path": path})
            for path, rule in json.loads(case["field_rules"]).items()
        ]
        grounded = any(
            rule.verified and any(ev.page is not None and ev.bbox is not None for ev in (rule.evidence or []))
            for rule in rules
        )
        assert row["grounding_eligible"] == str(grounded), row["id"]
        elapsed = float(row["elapsed_s"])
        cost = float(row["cost_usd"])
        assert math.isfinite(elapsed) and elapsed >= 0 and math.isfinite(cost) and cost >= 0, row["id"]
        assert int(row["pages"]) > 0, row["id"]
        late = elapsed > config["execution"]["leaderboard_deadline_s"]
        assert row["deadline_exceeded"] == str(late), row["id"]
        metrics = {}
        if row["original_outcome"] == "success" and not late:
            expected_predictions.add(row["id"])
            prediction = predictions[row["id"]]
            scored = compute_unified_evidence_metrics(
                json.loads(case["expected_output"]),
                prediction["extracted_data"],
                rules,
                prediction["field_citations"],
                schema,
                bbox_iou_threshold=config["evaluation"]["bbox_iou_threshold"],
                bbox_match_mode=config["evaluation"]["bbox_match_mode"],
            )
            metrics = {metric.metric_name: metric.value for metric in scored}
            assert METRICS["value_f1"] in metrics, row["id"]
            if grounded:
                assert all(METRICS[key] in metrics for key in ("word_grounding_f1", "page_grounding_f1")), row["id"]
        for column, metric in METRICS.items():
            if column != "value_f1" and not grounded:
                assert row[column] == "", row["id"]
            else:
                expected = metrics.get(metric, 0)
                assert math.isclose(float(row[column]), expected, rel_tol=0, abs_tol=1e-10), (row["id"], column)
    assert set(predictions) == expected_predictions, "Prediction coverage mismatch"
    assert sum(int(row["pages"]) for row in rows) == 2622
    assert sum(row["cost_sample"] == "original" for row in rows) == config["cost"]["original_measurements"]
    assert sum(row["cost_sample"] == "replacement" for row in rows) == config["cost"]["replacement_measurements"]
    assert math.isclose(sum(float(row["cost_usd"]) for row in rows), config["cost"]["total_usd"], abs_tol=1e-9)
    with (REPO / "leaderboard-pro.csv").open(newline="") as stream:
        leaderboard = [row for row in csv.DictReader(stream) if row["Provider"] == config["provider"]]
    expected_cohorts = {"all", *config["dataset"]["split_sha256"]}
    assert len(leaderboard) == len(expected_cohorts) and {row["Challenge"] for row in leaderboard} == expected_cohorts
    for entry in leaderboard:
        assert entry["Category"] == "Coding Agents"
        selected = [row for row in rows if entry["Challenge"] == "all" or row["id"].split("/")[0] == entry["Challenge"]]
        for column, expected in aggregate(selected).items():
            assert entry[column] == expected, (entry["Challenge"], column, entry[column], expected)
    print(f"Verified {len(rows)} documents, {len(predictions)} predictions, and {len(leaderboard)} leaderboard rows.")
    print(f"Estimated sample cost: ${config['cost']['total_usd']:.7f}; see README.md for accounting assumptions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot", type=Path, required=True, help="Pinned ExtractBenchPro snapshot with six JSONL files"
    )
    verify(parser.parse_args().snapshot)
