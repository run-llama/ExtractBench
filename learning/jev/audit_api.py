"""Offline Jev model-resolution and spend audit; emits no request/source payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent


def ledger_rows(directory):
    with sqlite3.connect(f"file:{directory / 'budget.sqlite'}?mode=ro", uri=True) as db:
        return db.execute("SELECT id,status,cost,tokens,request_hash FROM calls ORDER BY id").fetchall()


def safe_identifier(value):
    # Model/provider metadata is a string in the API contract. Never stringify
    # unexpected response objects: those could contain arbitrary private data.
    return value if isinstance(value, str) and len(value) <= 200 else "<missing-or-invalid>"


def decimal_value(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return Decimal(str(value))


def audit(directory):
    rows = ledger_rows(directory)
    requested, resolved, providers, status_codes = Counter(), Counter(), Counter(), Counter()
    accounting = defaultdict(lambda: {"attempts": 0, "accounted_usd": Decimal(0), "input_tokens": 0})
    missing, malformed, mismatched, usage_mismatches = [], [], [], []
    archive_hashes, metadata_missing = {}, Counter()
    archive_usage_cost, archive_input_tokens, archive_output_tokens = Decimal(0), 0, 0
    usage_cost_count, error_archive_count, decoded_archives = 0, 0, 0
    ledger_ids = {row[0] for row in rows}
    for call_id, status, cost, tokens, request_hash in rows:
        group = accounting[status]
        group["attempts"] += 1
        group["accounted_usd"] += Decimal(str(cost))
        group["input_tokens"] += tokens
        path = directory / f"call_{call_id:07d}.json"
        if not path.exists():
            missing.append(call_id)
            continue
        raw = path.read_bytes()
        archive_hashes[path.name] = hashlib.sha256(raw).hexdigest()
        try:
            record = json.loads(raw)
            if not isinstance(record, dict):
                raise ValueError("Archive must be an object")
        except (ValueError, UnicodeError):
            malformed.append(call_id)
            continue
        decoded_archives += 1
        request, response = record.get("request"), record.get("response")
        request = request if isinstance(request, dict) else {}
        response = response if isinstance(response, dict) else {}
        requested[safe_identifier(request.get("model"))] += 1
        resolved[safe_identifier(response.get("model"))] += 1
        providers[safe_identifier(response.get("provider"))] += 1
        status_code = record.get("status_code")
        status_codes[str(status_code) if isinstance(status_code, int) else "<missing>"] += 1
        if record.get("error") or response.get("error"):
            error_archive_count += 1
        for field in ("model", "provider", "id"):
            if not response.get(field):
                metadata_missing[field] += 1
        encoded = json.dumps(request, ensure_ascii=False).encode()
        if hashlib.sha256(encoded).hexdigest() != request_hash:
            mismatched.append(call_id)
        usage = response.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        usage_cost = decimal_value(usage.get("cost"))
        if usage_cost is not None:
            archive_usage_cost += usage_cost
            usage_cost_count += 1
            if status == "settled" and abs(usage_cost - Decimal(str(cost))) > Decimal("0.000000000001"):
                usage_mismatches.append(call_id)
        for field in ("input_tokens", "output_tokens"):
            value = usage.get(field)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                if field == "input_tokens":
                    archive_input_tokens += value
                else:
                    archive_output_tokens += value
    extras = sorted(
        int(path.stem.removeprefix("call_"))
        for path in directory.glob("call_*.json")
        if path.stem.removeprefix("call_").isdigit() and int(path.stem.removeprefix("call_")) not in ledger_ids
    )
    for group in accounting.values():
        group["accounted_usd"] = float(group["accounted_usd"])
    changed = rows != ledger_rows(directory)
    budget_caps = {9.0}
    for amendment in (directory.parent / "operational_resumes").glob("**/amendment.json"):
        value = json.loads(amendment.read_text()).get("new_cap_usd")
        if isinstance(value, (int, float)) and value >= 9.0:
            budget_caps.add(float(value))
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "scope": "All calls in the shared experiment ledger, including development, diagnostics, and full inference.",
        "ledger_attempts": len(rows),
        "max_call_id": max(ledger_ids, default=0),
        "requested_model_counts": dict(sorted(requested.items())),
        "resolved_model_counts": dict(sorted(resolved.items())),
        "response_provider_counts": dict(sorted(providers.items())),
        "http_status_counts": dict(sorted(status_codes.items())),
        "ledger_accounting_by_status": dict(accounting),
        "total_accounted_usd": float(sum((Decimal(str(row[2])) for row in rows), Decimal(0))),
        "initial_shared_budget_cap_usd": 9.0,
        "shared_budget_cap_usd": max(budget_caps),
        "documented_budget_caps_usd": sorted(budget_caps),
        "archive_usage": {
            "responses_with_cost": usage_cost_count,
            "reported_cost_usd": float(archive_usage_cost),
            "input_tokens": archive_input_tokens,
            "output_tokens": archive_output_tokens,
        },
        "preledger_smoke": {
            "successful_probe_cost_usd": 0.000014532,
            "included_in_ledger": False,
            "included_in_archive_usage": False,
        },
        "archive_completeness": {
            "decoded_archives": decoded_archives,
            "missing_call_ids": missing,
            "malformed_call_ids": malformed,
            "orphan_archive_call_ids": extras,
            "request_hash_mismatch_call_ids": mismatched,
            "settled_cost_mismatch_call_ids": usage_mismatches,
            "error_archives": error_archive_count,
            "missing_response_metadata_counts": dict(metadata_missing),
            "ledger_changed_during_audit": changed,
            "complete_and_consistent": not any((missing, malformed, extras, mismatched, usage_mismatches, changed)),
        },
        "archive_hash_manifest_sha256": hashlib.sha256(
            json.dumps(archive_hashes, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "archive_hash_method": "SHA256 of compact sorted JSON mapping archive basename to SHA256(file bytes).",
        "notes": [
            "Reserved costs are conservative accounting for unresolved requests, not confirmed provider charges.",
            "Model-resolution counts use returned response.model; HTTP failures can lack model/provider metadata.",
            "Usage sums cover archived responses only; no credit-balance request or other network call was made.",
            "No request state, question criteria, response answers, document text, "
            "credentials, or error bodies are emitted.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-dir", type=Path, default=HERE / "api")
    parser.add_argument("--output", type=Path, default=HERE / "final/api_audit.json")
    parser.add_argument(
        "--require-complete", action="store_true", help="Fail if archives/ledger are incomplete or changing"
    )
    args = parser.parse_args()
    report = audit(args.api_dir)
    if args.require_complete and not report["archive_completeness"]["complete_and_consistent"]:
        parser.error("API audit is incomplete/inconsistent; wait for workers or inspect archive counts")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "attempts": report["ledger_attempts"],
                "accounted_usd": report["total_accounted_usd"],
                "resolved_models": report["resolved_model_counts"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
