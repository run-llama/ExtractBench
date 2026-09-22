"""Rescore saved predictions using a separately pinned upstream evaluator.

Run with PYTHONPATH=<upstream>/src:<parse-bench-dependency-target>.
No model calls; leaves original result files intact.
"""

import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean

from extract_bench.evaluation.evaluators.extract import ExtractEvaluator
from extract_bench.schemas.extract_output import ExtractOutput, FieldCitation
from extract_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from extract_bench.schemas.product import ProductType
from extract_bench.test_cases.loader import load_test_case

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def local_module(name):
    path = ROOT / "src/extract_bench/inference/providers/extract/jev" / (name + ".py")
    spec = importlib.util.spec_from_file_location("jev_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant")
    ap.add_argument("--document")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    args = ap.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        ap.error("Require shard-count >= 1 and 0 <= shard-index < shard-count")
    citation_builder, parser = local_module("citations"), local_module("parser")
    summary = {}
    for directory in sorted((HERE / "results").iterdir()):
        if not directory.is_dir() or (args.variant and args.variant != directory.name):
            continue
        records = []
        for path in sorted(directory.glob("*.json")):
            d = json.loads(path.read_text())
            if "metrics" not in d or (args.document and args.document != d["document"]["id"]):
                continue
            row, prediction = d["document"], d.get("prediction")
            shard = int.from_bytes(hashlib.sha256(row["id"].encode("utf-8")).digest(), "big") % args.shard_count
            if shard != args.shard_index:
                continue
            out = HERE / "upstream" / directory.name / path.name
            out.parent.mkdir(parents=True, exist_ok=True)
            source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            citation_hash = hashlib.sha256(Path(citation_builder.__file__).read_bytes()).hexdigest()
            if out.exists():
                old = json.loads(out.read_text())
                if old.get("input_sha256") == source_hash and old.get("citation_sha256") == citation_hash:
                    records.append(old)
                    continue
            record = {
                "document": row,
                "variant": directory.name,
                "input_sha256": source_hash,
                "citation_sha256": citation_hash,
                "cost": d.get("cost", 0),
                "elapsed_seconds": d.get("elapsed_seconds", 0),
            }
            try:
                if not prediction or d.get("error"):
                    raise ValueError(d.get("error", "No completed prediction"))
                pdf = ROOT / "data" / row["split"] / (row["id"] + ".pdf")
                schema = json.loads(pdf.with_suffix(".test.json").read_text())["data_schema"]
                document = parser.parse_document(pdf, HERE / "parse_cache")
                citations = citation_builder.build_citations(prediction, document)
                now = datetime.fromisoformat(d["started_at"])
                result = InferenceResult(
                    request=InferenceRequest(
                        example_id=row["id"],
                        source_file_path=str(pdf),
                        product_type=ProductType.EXTRACT,
                        schema_override=schema,
                    ),
                    pipeline_name="jev_liteparse_" + directory.name,
                    product_type=ProductType.EXTRACT,
                    raw_output={
                        "prediction": prediction,
                        "cost_usd": d.get("cost", 0),
                        "num_pages": len(document["pages"]),
                        "cost_per_page_usd": d.get("cost", 0) / max(1, len(document["pages"])),
                        "num_api_calls": d.get("calls", 0),
                    },
                    output=ExtractOutput(
                        example_id=row["id"],
                        pipeline_name="jev_liteparse_" + directory.name,
                        extracted_data=prediction["data"],
                        field_citations=[FieldCitation(**c) for c in citations],
                    ),
                    started_at=now,
                    completed_at=now + timedelta(seconds=d.get("elapsed_seconds", 0)),
                    latency_in_ms=int(d.get("elapsed_seconds", 0) * 1000),
                )
                native = out.with_suffix(".result.json")
                native.write_text(result.model_dump_json())
                evaluation = ExtractEvaluator().evaluate(result, load_test_case(pdf, product_type_hint="extract"))
                record["metrics"] = {m.metric_name: m.value for m in evaluation.metrics}
                record["evaluation"] = evaluation.model_dump(mode="json")
                record["citations"] = len(citations)
            except Exception as exc:
                record["error"] = str(exc)
                record["metrics"] = {"extract_unified_value_f1": 0.0}
            out.write_text(json.dumps(record, ensure_ascii=False))
            records.append(record)
            print(
                directory.name,
                row["id"],
                round(record["metrics"]["extract_unified_value_f1"], 4),
                record.get("error", ""),
                flush=True,
            )
        groups = {}
        for split in ("all", "short", "medium", "long"):
            group = [r for r in records if split == "all" or r["document"]["split"] == split]
            if group:
                groups[split] = {
                    "n": len(group),
                    "f1": mean(r["metrics"].get("extract_unified_value_f1", 0) for r in group),
                    "failures": sum("error" in r for r in group),
                    "cost": sum(r.get("cost", 0) for r in group),
                    "word_f1": mean(
                        r["metrics"]["extract_unified_grounded_f1"]
                        for r in group
                        if "extract_unified_grounded_f1" in r["metrics"]
                    )
                    if any("extract_unified_grounded_f1" in r["metrics"] for r in group)
                    else None,
                }
        summary[directory.name] = groups
        print(directory.name, groups, flush=True)
    summary_name = (
        "upstream_summary.json"
        if args.shard_count == 1
        else f"upstream_summary.shard-{args.shard_index}-of-{args.shard_count}.json"
    )
    (HERE / summary_name).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
