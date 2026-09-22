"""Reproducible Jev experiments; persist predictions before loading labels."""

import argparse
import fcntl
import hashlib
import importlib
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from extract_bench.inference.providers.extract.jev.client import JevClient
from extract_bench.inference.providers.extract.jev.parser import parse_document

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODEL = "typesafe/jev-1.13"


def implementation_hashes():
    sources = list((ROOT / "src/extract_bench/inference/providers/extract/jev").glob("*.py"))
    sources += [
        Path(__file__).resolve(),
        ROOT / "src/extract_bench/inference/providers/extract/table_codegen/schema_utils.py",
        ROOT / "src/extract_bench/inference/providers/extract/jev_provider.py",
    ]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)}


def document_hashes(row):
    pdf = ROOT / "data" / row["split"] / (row["id"] + ".pdf")
    schema = json.loads(pdf.with_suffix(".test.json").read_text())["data_schema"]
    return {
        "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "schema_sha256": hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest(),
    }


def validate_freeze(freeze, variant, rows):
    if freeze.get("selected_variant") != variant or freeze.get("model") != MODEL:
        raise ValueError("Frozen variant/model does not match requested inference")
    expected = {
        str(Path(name).resolve().relative_to(ROOT)) if Path(name).is_absolute() else name: digest
        for name, digest in freeze.get("source_sha256", {}).items()
    }
    actual = implementation_hashes()
    for name, digest in actual.items():
        if expected.get(name) != digest:
            raise ValueError(f"Frozen implementation missing or changed: {name}")
    for name, digest in expected.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen source changed: {name}")
    frozen_documents = freeze.get("documents", {})
    for row in rows:
        actual_document = document_hashes(row)
        if frozen_documents.get(row["id"]) != actual_document:
            raise ValueError(f"Frozen document/schema missing or changed: {row['id']}")


def atomic_json(path, value):
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False))
    temporary.replace(path)


class DocumentClient(JevClient):
    """Bound paid calls at request boundaries; no automatic quality retries."""

    def __init__(self, *args, max_calls=2000, max_seconds=1800, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_calls, self.deadline = max_calls, time.monotonic() + max_seconds

    def _request(self, state, questions):
        if self.calls >= self.max_calls:
            raise RuntimeError(f"Per-document Jev call limit reached ({self.max_calls})")
        if time.monotonic() >= self.deadline:
            raise RuntimeError("Per-document Jev time limit reached")
        return super()._request(state, questions)


def evaluate_saved(record, pdf, schema, out):
    # Import and load ground truth only after a prediction is durably persisted.
    from extract_bench.evaluation.evaluators.extract import ExtractEvaluator
    from extract_bench.schemas.extract_output import ExtractOutput, FieldCitation
    from extract_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
    from extract_bench.schemas.product import ProductType
    from extract_bench.test_cases.loader import load_test_case

    name, variant = record["document"]["id"], record["variant"]
    citations = []
    for evidence in record["prediction"].get("evidence", []):
        if isinstance(evidence, dict) and "field_path" in evidence and "page" in evidence:
            citations.append(FieldCitation(**evidence))
    request = InferenceRequest(
        example_id=name, source_file_path=str(pdf), product_type=ProductType.EXTRACT, schema_override=schema
    )
    started = datetime.fromisoformat(record["started_at"])
    completed = datetime.fromisoformat(record.get("inference_completed_at", record["started_at"]))
    result = InferenceResult(
        request=request,
        pipeline_name="jev_" + variant,
        product_type=ProductType.EXTRACT,
        raw_output=record["prediction"],
        output=ExtractOutput(
            example_id=name,
            pipeline_name="jev_" + variant,
            extracted_data=record["prediction"]["data"],
            field_citations=citations,
        ),
        started_at=started,
        completed_at=completed,
        latency_in_ms=int(record.get("elapsed_seconds", (completed - started).total_seconds()) * 1000),
    )
    record["result"] = result.model_dump(mode="json")
    atomic_json(out, record)
    evaluation = ExtractEvaluator().evaluate(result, load_test_case(pdf, product_type_hint="extract"))
    record["evaluation"] = evaluation.model_dump(mode="json")
    record["metrics"] = {m.metric_name: m.value for m in evaluation.metrics}


def run_one(row, variant, *, max_calls=2000, max_seconds=1800):
    name = row["id"]
    out = HERE / "results" / variant / (name + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix(".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("BUSY", variant, name, flush=True)
            return
        _run_locked(row, variant, out, max_calls, max_seconds)


def _run_locked(row, variant, out, max_calls, max_seconds):
    name = row["id"]
    record = json.loads(out.read_text()) if out.exists() else None
    # Legacy final results and errors are final: never pay to retry them for quality.
    if record is not None and (record.get("completed") or "metrics" in record or "error" in record):
        record["completed"] = True
        record.setdefault("metrics", {"extract_unified_value_f1": 0.0})
        atomic_json(out, record)
        print("REUSE", variant, name, flush=True)
        return
    pdf = ROOT / "data" / row["split"] / (name + ".pdf")
    schema = json.loads(pdf.with_suffix(".test.json").read_text())["data_schema"]
    hashes = document_hashes(row)
    if row.get("pdf_sha256", hashes["pdf_sha256"]) != hashes["pdf_sha256"]:
        raise ValueError(f"Manifest PDF changed: {name}")
    if row.get("schema_hash", hashes["schema_sha256"]) != hashes["schema_sha256"]:
        raise ValueError(f"Manifest schema changed: {name}")
    started = datetime.now()
    if record is None:
        source_hashes = implementation_hashes()
        record = {
            "document": row,
            "variant": variant,
            "source_sha256": source_hashes,
            "artifact_sha256": hashes,
            "model": MODEL,
            "started_at": started.isoformat(),
            "completed": False,
            "calls": 0,
            "cost": 0.0,
        }
        atomic_json(out, record)
    elif "prediction" not in record:
        # An interrupted inference may already have incurred cost. Preserve failure
        # rather than issuing an implicit paid rerun of an unknown partial attempt.
        record.update(
            error="Interrupted inference without persisted prediction; not automatically retried",
            completed=True,
            metrics={"extract_unified_value_f1": 0.0},
        )
        atomic_json(out, record)
        print("INTERRUPTED", variant, name, flush=True)
        return
    print("RESUME_EVALUATION" if "prediction" in record else "START", variant, row["split"], name, flush=True)
    client = None
    try:
        if "prediction" not in record:
            module = importlib.import_module("extract_bench.inference.providers.extract.jev.variant_" + variant)
            client = DocumentClient(
                HERE / "api", label=f"{variant}/{name}", model=MODEL, max_calls=max_calls, max_seconds=max_seconds
            )
            document = parse_document(pdf, HERE / "parse_cache")
            record["parser_version"] = document["parser_version"]
            record["prediction"] = module.extract(document, schema, client)
            record.update(
                calls=client.calls,
                cost=client.cost,
                inference_completed_at=datetime.now().isoformat(),
                elapsed_seconds=(datetime.now() - started).total_seconds(),
            )
            atomic_json(out, record)
        evaluate_saved(record, pdf, schema, out)
    except Exception as exc:
        record.update(error=str(exc), traceback=traceback.format_exc(), metrics={"extract_unified_value_f1": 0.0})
    if client is not None:
        record.update(calls=client.calls, cost=client.cost, elapsed_seconds=(datetime.now() - started).total_seconds())
    record["completed"] = True
    atomic_json(out, record)
    print(
        "DONE",
        variant,
        name,
        "F1",
        round(record["metrics"]["extract_unified_value_f1"], 4),
        "cost",
        round(record.get("cost", 0), 5),
        record.get("error", ""),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant")
    parser.add_argument("--partition", choices=["dev", "holdout", "full"], default="dev")
    parser.add_argument("--document")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-calls", type=int, default=2000)
    parser.add_argument("--max-seconds", type=float, default=1800)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    manifest = json.loads((HERE / "manifest.json").read_text())
    rows = [r for r in manifest["documents"] if r["partition"] == args.partition]
    if args.partition == "full":
        rows = json.loads((ROOT / "learning/luna_round2/inventory.json").read_text())
    if args.document:
        rows = [r for r in rows if r["id"] == args.document]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        parser.error("No matching documents")
    if args.max_calls <= 0 or args.max_seconds <= 0:
        parser.error("Per-document limits must be positive")
    if args.partition != "dev":
        freeze_path = HERE / "freeze.json"
        if not freeze_path.exists():
            parser.error("Freeze implementation before held-out or full evaluation")
        validate_freeze(json.loads(freeze_path.read_text()), args.variant, rows)
    if args.workers <= 1:
        for row in rows:
            run_one(row, args.variant, max_calls=args.max_calls, max_seconds=args.max_seconds)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(run_one, row, args.variant, max_calls=args.max_calls, max_seconds=args.max_seconds)
                for row in rows
            ]
            for future in as_completed(futures):
                future.result()


if __name__ == "__main__":
    main()
