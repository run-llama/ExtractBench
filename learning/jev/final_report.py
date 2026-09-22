"""Aggregate complete frozen Jev inference using the pinned upstream evaluator.

Run with PYTHONPATH=<upstream>/src:<dependency-target>. No network/model calls.
Unfinished or stale artifacts are fatal; failed inference remains in denominators.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VALUE = "extract_unified_value_"
WORD = "extract_unified_grounded_f1"
PAGE = "extract_unified_page_f1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def failure_native(row, variant, prediction):
    """Build a transport placeholder; official runner failures score zero."""
    from extract_bench.schemas.extract_output import ExtractOutput
    from extract_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
    from extract_bench.schemas.product import ProductType

    pdf = ROOT / "data" / row["split"] / (row["id"] + ".pdf")
    schema = read(pdf.with_suffix(".test.json"))["data_schema"]
    started = datetime.fromisoformat(prediction["started_at"])
    completed = datetime.fromisoformat(
        prediction.get("operational_resume", {}).get("completed_at")
        or prediction.get("inference_completed_at")
        or prediction.get("completed_at")
        or prediction["started_at"]
    )
    result = InferenceResult(
        request=InferenceRequest(
            example_id=row["id"], source_file_path=str(pdf), product_type=ProductType.EXTRACT, schema_override=schema
        ),
        pipeline_name="jev_liteparse_" + variant,
        product_type=ProductType.EXTRACT,
        output=ExtractOutput(
            example_id=row["id"], pipeline_name="jev_liteparse_" + variant, extracted_data={}, field_citations=[]
        ),
        raw_output={"inference_failed": True},
        started_at=started,
        completed_at=completed,
        latency_in_ms=0,
    )
    return result.model_dump(mode="json")


def core_latency(record):
    try:
        start = datetime.fromisoformat(record["started_at"])
        end = datetime.fromisoformat(record["inference_completed_at"])
        seconds = (end - start).total_seconds()
        return seconds if seconds >= 0 else None
    except (KeyError, TypeError, ValueError):
        return None


def aggregate(rows):
    if not rows:
        return {"documents": 0}
    result = {
        "documents": len(rows),
        "failures": sum(row["failed"] for row in rows),
        "budget_resumed_documents": sum(row.get("budget_resumed", False) for row in rows),
        "replayed_api_requests": sum(row.get("replayed_api_requests", 0) for row in rows),
        "value_precision": mean(row["precision"] for row in rows),
        "value_recall": mean(row["recall"] for row in rows),
        "value_f1": mean(row["f1"] for row in rows),
        "total_reported_api_cost_usd": sum(row["cost_usd"] for row in rows),
        "mean_cost_per_document_usd": mean(row["cost_usd"] for row in rows),
        "mean_document_cost_per_page_usd": mean(row["cost_per_page_usd"] for row in rows),
        "total_pages": sum(row["pages"] for row in rows),
    }
    for label in ("word", "page"):
        eligible = [row for row in rows if row[f"{label}_f1"] is not None]
        result[f"{label}_grounding_eligible_documents"] = len(eligible)
        result[f"{label}_grounding_failed_eligible_documents"] = sum(row["failed"] for row in eligible)
        result[f"{label}_grounding_f1"] = mean(row[f"{label}_f1"] for row in eligible) if eligible else None
    latencies = [row["core_inference_seconds"] for row in rows if row["core_inference_seconds"] is not None]
    result["core_latency_measured_documents"] = len(latencies)
    result["mean_core_inference_seconds"] = mean(latencies) if latencies else None
    return result


def group_report(rows):
    return {
        split: aggregate([r for r in rows if split == "all" or r["split"] == split])
        for split in ("all", "short", "medium", "long")
    }


def leaderboard_row(groups, columns, provider):
    row = dict.fromkeys(columns, "")
    row.update(Provider=provider, Category="Specialized APIs")
    for split, suffix in (("all", ""), ("short", "_Short"), ("medium", "_Medium"), ("long", "_Long")):
        group = groups[split]
        row["Overall" if not suffix else suffix[1:]] = f"{group['value_f1'] * 100:.2f}"
        row["Cost_Per_Page" if not suffix else "Cost" + suffix] = f"{group['mean_document_cost_per_page_usd']:.4f}"
        for name, metric in (("Word_Grounding", "word_grounding_f1"), ("Page_Grounding", "page_grounding_f1")):
            value = group[metric]
            row[name + suffix] = f"{value * 100:.2f}" if value is not None else ""
        if suffix:
            row["P" + suffix] = f"{group['value_precision'] * 100:.2f}"
            row["R" + suffix] = f"{group['value_recall'] * 100:.2f}"
    # Cold end-to-end provider latency cannot be reconstructed from warm parse
    # caches and evaluation-inclusive historical clocks. Do not publish it.
    return row


def sanitized_native(native, row, variant):
    result = dict(native)
    request = dict(result.get("request", {}))
    request["source_file_path"] = f"data/{row['split']}/{row['id']}.pdf"
    request["example_id"] = f"{row['split']}/{row['id']}"
    # Retain the public schema and extracted output, discard source/debug payloads.
    result["request"] = request
    if isinstance(result.get("output"), dict):
        result["output"] = {**result["output"], "example_id": request["example_id"]}
    result["pipeline_name"] = "jev_liteparse_" + variant
    result["raw_output"] = {
        "cost_usd": row["cost_usd"],
        "cost_per_page_usd": row["cost_per_page_usd"],
        "num_pages": row["pages"],
        "inference_failed": row["failed"],
        "budget_resumed": row["budget_resumed"],
        "replayed_api_requests": row["replayed_api_requests"],
        "latency_scope": "warm-cache core inference; excludes citation conversion and evaluator",
    }
    if row["core_inference_seconds"] is not None:
        result["latency_in_ms"] = int(row["core_inference_seconds"] * 1000)
    else:
        result["latency_in_ms"] = 0
        result["raw_output"]["latency_unavailable"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="router")
    parser.add_argument("--output", type=Path, default=HERE / "final")
    parser.add_argument("--leaderboard", type=Path, default=ROOT / "leaderboard.csv")
    parser.add_argument("--provider-name", default="Jev 1.13 + LiteParse")
    args = parser.parse_args()
    freeze_path = HERE / "freeze.json"
    if not freeze_path.exists():
        parser.error("Missing freeze.json; no final report can be produced")
    freeze = read(freeze_path)
    if freeze.get("selected_variant") != args.variant:
        parser.error("Requested variant differs from frozen selected_variant")
    inventory = read(ROOT / "learning/luna_round2/inventory.json")
    if len(inventory) != 370 or len({row["id"] for row in inventory}) != 370:
        parser.error("Expected exactly 370 unique official corpus documents")
    pending, sources = [], []
    for row in inventory:
        prediction_path = HERE / "results" / args.variant / (row["id"] + ".json")
        score_path = HERE / "upstream" / args.variant / (row["id"] + ".json")
        if not prediction_path.exists() or not score_path.exists():
            pending.append(row["id"])
            continue
        prediction, scored = read(prediction_path), read(score_path)
        if not prediction.get("completed") or "metrics" not in scored:
            pending.append(row["id"])
            continue
        if prediction.get("source_sha256") != freeze.get("source_sha256"):
            parser.error(f"Prediction source hashes differ from frozen implementation: {row['id']}")
        if prediction.get("artifact_sha256") != freeze.get("documents", {}).get(row["id"]):
            parser.error(f"Prediction document/schema hashes differ from freeze: {row['id']}")
        if prediction.get("model") != freeze.get("model"):
            parser.error(f"Prediction model differs from freeze: {row['id']}")
        if scored.get("input_sha256") != sha(prediction_path):
            parser.error(f"Stale upstream scoring for {row['id']}; rerun rescore.py")
        if scored.get("citation_sha256") != sha(
            ROOT / "src/extract_bench/inference/providers/extract/jev/citations.py"
        ):
            parser.error(f"Stale citation conversion for {row['id']}; rerun rescore.py")
        sources.append((row, prediction, scored, prediction_path, score_path))
    if pending:
        parser.error(
            f"Full report requires all 370 final predictions and matching upstream scores; "
            f"{len(pending)} unfinished. First IDs: {', '.join(pending[:8])}"
        )
    evaluator = read(HERE / "evaluator.json")
    import extract_bench.evaluation.evaluators.extract as evaluator_module

    expected_evaluator = Path(evaluator["upstream_checkout"]) / "src/extract_bench/evaluation/evaluators/extract.py"
    if Path(evaluator_module.__file__).resolve() != expected_evaluator.resolve():
        parser.error("Run with PYTHONPATH=<pinned-upstream>/src:<dependency-target>; local evaluator is not accepted")
    dataset = read(HERE / "dataset_verification.json")
    if not dataset.get("all_components_match"):
        parser.error("Dataset integrity verification is incomplete or mismatched")
    if dataset.get("remote_revision") != freeze.get("dataset_revision"):
        parser.error("Verified dataset revision differs from freeze")
    rows, natives, input_hashes, operational_resumes = [], [], {}, {}
    for document, prediction, scored, prediction_path, score_path in sources:
        failed = bool(prediction.get("error") or scored.get("error"))
        metrics = scored["metrics"]
        if failed:
            # EvaluationRunner._aggregate_metrics pads every observed score
            # family with one zero per failed extraction, regardless of the
            # failed document's annotation coverage. Match that public runner.
            word_eligible, page_eligible = True, True
            native = failure_native(document, args.variant, prediction)
        else:
            for key in (VALUE + "precision", VALUE + "recall", VALUE + "f1"):
                if key not in metrics:
                    parser.error(f"Missing official {key} for successful {document['id']}")
            word_eligible, page_eligible = WORD in metrics, PAGE in metrics
            native_path = score_path.with_suffix(".result.json")
            if not native_path.exists():
                parser.error(f"Missing native inference artifact for {document['id']}")
            native = read(native_path)
        pages = int(document["pages"])
        if pages <= 0:
            parser.error(f"Invalid page denominator for {document['id']}")
        cost = float(prediction.get("cost", 0))
        record = {
            "id": document["id"],
            "split": document["split"],
            "pages": pages,
            "failed": failed,
            "precision": 0.0 if failed else metrics[VALUE + "precision"],
            "recall": 0.0 if failed else metrics[VALUE + "recall"],
            "f1": 0.0 if failed else metrics[VALUE + "f1"],
            "word_f1": (0.0 if failed else metrics[WORD]) if word_eligible else None,
            "page_f1": (0.0 if failed else metrics[PAGE]) if page_eligible else None,
            "cost_usd": cost,
            "cost_per_page_usd": cost / pages,
            "core_inference_seconds": core_latency(prediction),
            "api_calls": prediction.get("calls", 0),
            "budget_resumed": bool(prediction.get("operational_resume")),
            "replayed_api_requests": prediction.get("operational_resume", {}).get("replayed_successes", 0),
        }
        if prediction.get("operational_resume"):
            # Explicit allowlist: no source text, request bodies, or answers.
            allowed = {
                "amendment_path",
                "original_calls",
                "original_cost_usd",
                "original_record_sha256",
                "wrapper_sha256",
                "operational_cap_usd",
                "deadline_policy",
                "new_calls",
                "new_accounted_cost_usd",
                "replayed_successes",
                "consumed_old_attempts",
                "elapsed_seconds",
                "completed_at",
                "latency_scope",
            }
            operational_resumes[document["id"]] = {
                key: value for key, value in prediction["operational_resume"].items() if key in allowed
            }
            operational_resumes[document["id"]]["resume_failed"] = bool(
                prediction["operational_resume"].get("resume_error")
            )
        rows.append(record)
        natives.append(sanitized_native(native, record, args.variant))
        input_hashes[document["id"]] = {"prediction_sha256": sha(prediction_path), "score_sha256": sha(score_path)}
    manifest = read(HERE / "manifest.json")
    holdout_ids = {row["id"] for row in manifest["documents"] if row["partition"] == "holdout"}
    if len(holdout_ids) != 12:
        parser.error("Expected 12 holdout documents in experiment manifest")
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "variant": args.variant,
        "full": group_report(rows),
        "holdout": group_report([row for row in rows if row["id"] in holdout_ids]),
        "methodology": {
            "value": "Unweighted mean over all 370 documents; "
            "any failed document contributes zero precision/recall/F1.",
            "grounding": "Official unchanged EvaluationRunner macro aggregation: successful documents contribute "
            "only when they emit the corresponding grounding metric. Every failed extraction adds one zero "
            "to each pipeline-observed grounding metric, regardless of that failed document's annotation coverage. "
            "Successful grounded_incomplete cases remain excluded. Reported denominators include all failed documents.",
            "cost": "Mean of each document's reported API cost divided by its page count; "
            "not total cost divided by total pages. "
            "Excludes local compute and prior development experiments. "
            "Unresolved reservations may exceed actual charges.",
            "latency": "Core inference_completed_at minus started_at where available: "
            "warm cached LiteParse parsing plus inference; "
            "excludes citation construction and evaluation. Missing timestamps excluded from latency mean only. "
            "Leaderboard latency columns intentionally blank; cold end-to-end latency not measured.",
            "overlap": "Full 370 includes 12 development documents. Prior Luna study saw designated holdout labels; "
            "holdout is for Jev selection, not previously unseen in the broader project.",
            "operational_budget_continuation": "Only shared-budget-interrupted documents were eligible. "
            "The cumulative study cap increased from $9.00 to $9.75 because development consumed the shared ledger. "
            "Exact ordered archived request hashes and successful answers were replayed before any new request; "
            "successful document outputs were never rerolled. Original charges and failed-request reservations "
            "remain counted once, with new charges added. Resume receives a fresh 1800-second allowance, "
            "while the combined original-plus-new attempt limit remains 2000. The standalone framework's "
            "default $9 budget and frozen inference code are unchanged. Resumed core latency is unavailable.",
        },
        "provenance": {
            "freeze": freeze,
            "freeze_sha256": sha(freeze_path),
            "evaluator": evaluator,
            "dataset_verification": dataset,
            "input_artifacts": input_hashes,
            "operational_resumes": operational_resumes,
        },
    }
    ledger = HERE / "api/budget.sqlite"
    if ledger.exists():
        with sqlite3.connect(f"file:{ledger}?mode=ro", uri=True) as db:
            report["ledger"] = {
                "scope_note": "Ledger labels include archived development runs. "
                "Full-corpus cost comes from the 370 final per-document artifacts, not this all-runs ledger subtotal.",
                "all_experiments": [
                    {"status": status, "attempts": n, "accounted_usd": cost}
                    for status, n, cost in db.execute("SELECT status,COUNT(*),SUM(cost) FROM calls GROUP BY status")
                ],
                "selected_variant_all_runs_including_development": [
                    {"status": status, "attempts": n, "accounted_usd": cost}
                    for status, n, cost in db.execute(
                        "SELECT status,COUNT(*),SUM(cost) FROM calls WHERE label LIKE ? GROUP BY status",
                        (args.variant + "/%",),
                    )
                ],
            }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "per_document.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with args.leaderboard.open(newline="") as handle:
        columns = next(csv.reader(handle))
    with (args.output / "proposed_leaderboard_row.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow(leaderboard_row(report["full"], columns, args.provider_name))
    # Stable gzip timestamp makes a second report run byte-reproducible.
    with (args.output / "native_results.jsonl.gz").open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as handle:
                for native in natives:
                    handle.write(json.dumps(native, ensure_ascii=False, allow_nan=False) + "\n")
    report["artifact_sha256"] = {
        name: sha(args.output / name)
        for name in ("per_document.csv", "proposed_leaderboard_row.csv", "native_results.jsonl.gz")
    }
    atomic_json(args.output / "report.json", report)
    print(json.dumps({"full": report["full"], "holdout": report["holdout"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
