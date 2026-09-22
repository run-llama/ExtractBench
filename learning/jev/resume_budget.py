"""Optional budget-only continuation using exact archived response replay.

Dry-run by default. --execute is an explicit operational action, allowed only
once original workers stop. Never retries a successful or non-budget failure.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib
import json
import os
import re
import sqlite3
import time
import traceback
from datetime import datetime
from pathlib import Path

import run as runner
from dotenv import load_dotenv

HERE, ROOT = runner.HERE, runner.ROOT
CAP = 9.75
BUDGET_ERROR = re.compile(r"Shared Jev budget exhausted: \$\d+\.\d{4}/\$9\.00")


def request_hash(body):
    return hashlib.sha256(json.dumps(body, ensure_ascii=False).encode()).hexdigest()


def immutable_json(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError("Immutable operational artifact already exists with different contents")
        return
    with path.open("xb") as handle:
        handle.write(raw)
    path.chmod(0o444)


def prefix_for_record(directory, record):
    count = int(record["calls"])
    if count < 0 or count > 2000:
        raise ValueError("Invalid original call count")
    label = f"router/{record['document']['id']}"
    with sqlite3.connect(f"file:{directory / 'budget.sqlite'}?mode=ro", uri=True) as db:
        rows = db.execute(
            "SELECT id,status,cost,request_hash FROM calls WHERE label=? ORDER BY id DESC LIMIT ?", (label, count)
        ).fetchall()[::-1]
    if len(rows) != count or abs(sum(row[2] for row in rows) - record["cost"]) > 1e-9:
        raise ValueError("Original call-count/cost does not match latest per-document ledger prefix")
    prefix = []
    for call_id, status, cost, digest in rows:
        path = directory / f"call_{call_id:07d}.json"
        archive = json.loads(path.read_text())
        body = archive.get("request", {})
        if archive.get("label") != label or request_hash(body) != digest or body.get("model") != record["model"]:
            raise ValueError(f"Archive/request/label/model mismatch at call {call_id}")
        response = archive.get("response", {})
        success = archive.get("status_code") == 200 and not archive.get("error")
        if success:
            if not isinstance(response, dict) or set(response.get("answers", {})) != set(body.get("questions", {})):
                raise ValueError(f"Successful archive has missing answers at call {call_id}")
            answers = response["answers"]
        else:
            if archive.get("status_code") not in (None, 429, 500, 502, 503, 504) or not archive.get("error"):
                raise ValueError(f"Non-retryable failed attempt in budget-interrupted prefix at call {call_id}")
            answers = None
        prefix.append(
            {
                "id": call_id,
                "request_hash": digest,
                "answers": answers,
                "status": status,
                "cost": cost,
                "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return prefix


class ReplayClient(runner.DocumentClient):
    def __init__(self, directory, record, prefix):
        super().__init__(
            directory,
            label=f"router/{record['document']['id']}",
            model=record["model"],
            budget=CAP,
            max_calls=2000,
            max_seconds=1800,
        )
        self.calls, self.cost = record["calls"], record["cost"]
        self.prefix, self.position, self.replayed_successes = prefix, 0, 0
        self.new_calls_started = False

    def _request(self, state, questions):
        digest = request_hash({"model": self.model, "state": state, "questions": questions})
        while self.position < len(self.prefix):
            saved = self.prefix[self.position]
            if saved["request_hash"] != digest:
                raise ValueError(f"Replay request order/hash mismatch before new API call at old call {saved['id']}")
            self.position += 1
            if saved["answers"] is not None:
                self.replayed_successes += 1
                return copy.deepcopy(saved["answers"])
            # A failed transient attempt consumed budget but supplied no answer.
            # Its following retry must have exactly the same request hash.
        self.new_calls_started = True
        return super()._request(state, questions)


def require_idle(inventory):
    # An incomplete record also catches multiprocessing children whose command
    # line no longer names run.py. Locks cover the interval between checks.
    for row in inventory:
        path = HERE / "results/router" / (row["id"] + ".json")
        if not path.exists() or not json.loads(path.read_text()).get("completed"):
            raise ValueError("Original full-corpus workers must finish all records before budget continuation")
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            args = (process / "cmdline").read_bytes().split(b"\0")
        except (OSError, PermissionError):
            continue
        if any(argument.endswith(b"/learning/jev/run.py") or argument == b"learning/jev/run.py" for argument in args):
            raise ValueError("Original run.py process is still active")


def continue_one(path, record, prefix, freeze):
    row = record["document"]
    runner.validate_freeze(freeze, "router", [row])
    folder = HERE / "operational_resumes/budget_9_75" / hashlib.sha256(row["id"].encode()).hexdigest()[:20]
    original = path.read_bytes()
    immutable_json(folder / "original.json", original)
    if (folder / "amendment.json").exists():
        raise ValueError("This document already has a budget continuation; refusing an implicit additional attempt")
    provenance = {
        "document_id": row["id"],
        "amendment": "Shared study accounting cap9.00 ->9.75; frozen inference unchanged",
        "old_cap_usd": 9.0,
        "new_cap_usd": CAP,
        "original_record_sha256": hashlib.sha256(original).hexdigest(),
        "original_calls": record["calls"],
        "original_cost_usd": record["cost"],
        "freeze_sha256": hashlib.sha256((HERE / "freeze.json").read_bytes()).hexdigest(),
        "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "deadline_policy": "New 1800-second operational allowance starts at resume; old calls still count toward2000.",
        "prefix": [{k: v for k, v in item.items() if k != "answers"} for item in prefix],
        "started_at": datetime.now().isoformat(),
    }
    immutable_json(folder / "amendment.json", json.dumps(provenance, indent=2).encode())
    client = ReplayClient(HERE / "api", record, prefix)
    result = copy.deepcopy(record)
    for key in ("traceback", "metrics", "evaluation", "result", "prediction", "inference_completed_at"):
        result.pop(key, None)
    result.update(
        completed=False,
        metrics={"extract_unified_value_f1": 0.0},
        operational_resume={
            "amendment_path": str(folder.relative_to(ROOT) / "amendment.json"),
            "original_calls": record["calls"],
            "original_cost_usd": record["cost"],
            "original_record_sha256": provenance["original_record_sha256"],
            "wrapper_sha256": provenance["wrapper_sha256"],
            "operational_cap_usd": CAP,
            "deadline_policy": provenance["deadline_policy"],
        },
    )
    started = time.monotonic()
    try:
        pdf = ROOT / "data" / row["split"] / (row["id"] + ".pdf")
        schema = json.loads(pdf.with_suffix(".test.json").read_text())["data_schema"]
        document = runner.parse_document(pdf, HERE / "parse_cache")
        module = importlib.import_module("extract_bench.inference.providers.extract.jev.variant_router")
        result["prediction"] = module.extract(document, schema, client)
        if client.position != len(prefix):
            raise ValueError("Extraction completed without consuming its full archived request prefix")
        result.pop("error", None)
        result.update(calls=client.calls, cost=client.cost)
        runner.atomic_json(path, result)
        runner.evaluate_saved(result, pdf, schema, path)
    except Exception as exc:
        result.update(
            error=str(exc) if "prediction" in result else record["error"],
            traceback=traceback.format_exc(),
            metrics={"extract_unified_value_f1": 0.0},
        )
        result["operational_resume"]["resume_error"] = str(exc)
    finally:
        # Account old and new attempts exactly once, even if continuation fails.
        result.update(calls=client.calls, cost=client.cost, completed=True)
        result["operational_resume"].update(
            new_calls=client.calls - record["calls"],
            new_accounted_cost_usd=client.cost - record["cost"],
            replayed_successes=client.replayed_successes,
            consumed_old_attempts=client.position,
            elapsed_seconds=time.monotonic() - started,
            completed_at=datetime.now().isoformat(),
            latency_scope="Original and resumed attempts separated by idle time; core latency unavailable",
        )
        runner.atomic_json(path, result)
        runner.atomic_json(
            folder / "outcome.json",
            {
                "completed": True,
                "error": result.get("error"),
                "calls": client.calls,
                "cost": client.cost,
                "result_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                **result["operational_resume"],
            },
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document")
    parser.add_argument("--execute", action="store_true", help="Explicitly enable new paid continuation requests")
    args = parser.parse_args()
    freeze = json.loads((HERE / "freeze.json").read_text())
    inventory = json.loads((ROOT / "learning/luna_round2/inventory.json").read_text())
    if len(inventory) != 370:
        parser.error("Expected complete370-document inventory")
    require_idle(inventory)
    candidates = []
    for row in inventory:
        if args.document and row["id"] != args.document:
            continue
        path = HERE / "results/router" / (row["id"] + ".json")
        record = json.loads(path.read_text())
        if not BUDGET_ERROR.fullmatch(record.get("error", "")):
            continue
        if record.get("prediction") is not None or record.get("operational_resume"):
            parser.error(f"Refusing to resume a produced prediction or prior continuation: {row['id']}")
        runner.validate_freeze(freeze, "router", [row])
        if (
            record.get("source_sha256") != freeze.get("source_sha256")
            or record.get("artifact_sha256") != freeze.get("documents", {}).get(row["id"])
            or record.get("model") != freeze.get("model")
        ):
            parser.error(f"Frozen provenance mismatch: {row['id']}")
        candidates.append((path, record, prefix_for_record(HERE / "api", record)))
    print(
        json.dumps(
            {
                "dry_run": not args.execute,
                "eligible_documents": len(candidates),
                "operational_cap_usd": CAP,
                "old_calls": sum(record["calls"] for _, record, _ in candidates),
            },
            indent=2,
        )
    )
    if not args.execute:
        return
    load_dotenv(ROOT / ".env")
    with (HERE / "budget_resume.lock").open("a") as global_lock:
        fcntl.flock(global_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for path, record, prefix in candidates:
            with path.with_suffix(".lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if json.loads(path.read_text()) != record:
                    raise ValueError("Original record changed after continuation preflight")
                result = continue_one(path, record, prefix, freeze)
                print(
                    "RESUMED",
                    record["document"]["id"],
                    "calls",
                    result["calls"],
                    "cost",
                    result["cost"],
                    "failed",
                    bool(result.get("error")),
                    flush=True,
                )
                if result.get("error", "").startswith("Shared Jev budget exhausted:"):
                    break


if __name__ == "__main__":
    main()
