"""Optional parallel handoff of budget continuations; dry-run by default.

Uses the unchanged resume_budget.py wrapper and its exact archive replay.
Per-document locks are the ownership boundary. The sequential parent's global
lock is intentionally not acquired: its currently locked document is untouched.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import resume_budget as resume
from dotenv import load_dotenv

HERE, ROOT = resume.HERE, resume.ROOT


def artifact_folder(identifier):
    return HERE / "operational_resumes/budget_9_75" / hashlib.sha256(identifier.encode()).hexdigest()[:20]


def eligible(record):
    return (
        record.get("completed") is True
        and bool(resume.BUDGET_ERROR.fullmatch(record.get("error", "")))
        and record.get("prediction") is None
        and not record.get("operational_resume")
    )


def process_document(row, freeze, execute=False):
    """Claim before inspecting eligibility, so parent/coordinator cannot race."""
    path = HERE / "results/router" / (row["id"] + ".json")
    with path.with_suffix(".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"id": row["id"], "status": "busy_parent_or_other_worker"}
        record = json.loads(path.read_text())
        if not eligible(record):
            return {"id": row["id"], "status": "ineligible_or_already_finished"}
        if (artifact_folder(row["id"]) / "amendment.json").exists():
            return {"id": row["id"], "status": "existing_amendment_not_retried"}
        resume.runner.validate_freeze(freeze, "router", [row])
        if (
            record.get("source_sha256") != freeze.get("source_sha256")
            or record.get("artifact_sha256") != freeze.get("documents", {}).get(row["id"])
            or record.get("model") != freeze.get("model")
        ):
            raise ValueError(f"Frozen provenance mismatch: {row['id']}")
        prefix = resume.prefix_for_record(HERE / "api", record)
        if not execute:
            return {"id": row["id"], "status": "eligible_dry_run", "old_calls": record["calls"]}
        result = resume.continue_one(path, record, prefix, freeze)
        return {
            "id": row["id"],
            "status": "continued",
            "failed": bool(result.get("error")),
            "total_calls": result["calls"],
            "total_cost_usd": result["cost"],
            "new_calls": result["operational_resume"].get("new_calls", 0),
        }


def outcome_snapshot(inventory):
    outcomes, interrupted, originals, unfinished = [], [], [], []
    for row in inventory:
        folder = artifact_folder(row["id"])
        if (folder / "original.json").exists():
            originals.append(row["id"])
        if (folder / "outcome.json").exists():
            outcomes.append(row["id"])
        path = HERE / "results/router" / (row["id"] + ".json")
        if not path.exists():
            unfinished.append(row["id"])
            continue
        record = json.loads(path.read_text())
        if not record.get("completed"):
            unfinished.append(row["id"])
        if eligible(record):
            interrupted.append(row["id"])
    return {
        "outcome_count": len(outcomes),
        "outcome_ids": outcomes,
        "immutable_original_count": len(originals),
        "immutable_original_ids": originals,
        "unclaimed_or_parent_running_ids": interrupted,
        "unfinished_ids": unfinished,
    }


def active_sequential_parents():
    parents = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit() or int(path.name) == os.getpid():
            continue
        try:
            args = (path / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if any(arg.endswith(b"/resume_budget.py") or arg == b"resume_budget.py" for arg in args):
            parents.append(int(path.name))
        if any(arg.endswith(b"/learning/jev/run.py") or arg == b"learning/jev/run.py" for arg in args):
            raise ValueError("Original inference workers are still active; handoff is only for budget continuations")
    return parents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--expected-outcomes", type=int, default=18)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.expected_outcomes < 1 or (args.verify_only and args.execute):
        parser.error("Require positive worker/outcome counts; verify-only and execute are mutually exclusive")
    inventory = json.loads((ROOT / "learning/luna_round2/inventory.json").read_text())
    if len(inventory) != 370:
        parser.error("Expected all 370 original documents")
    freeze = json.loads((HERE / "freeze.json").read_text())
    resume.runner.validate_freeze(freeze, "router", [])
    parents = active_sequential_parents()
    initial = outcome_snapshot(inventory)
    if args.verify_only:
        print(json.dumps(initial, indent=2))
        if (
            initial["outcome_count"] != args.expected_outcomes
            or initial["immutable_original_count"] != args.expected_outcomes
            or initial["unfinished_ids"]
            or initial["unclaimed_or_parent_running_ids"]
        ):
            parser.error("Not all expected budget continuation outcomes are final yet")
        return
    if any(not (HERE / "results/router" / (row["id"] + ".json")).exists() for row in inventory):
        parser.error("Original inference has not produced all 370 records")
    identifier = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + f"-{os.getpid()}"
    report_path = HERE / "operational_resumes" / f"parallel_coordination-{identifier}.json"
    coordination = {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "execute": args.execute,
        "workers": args.workers,
        "expected_outcomes": args.expected_outcomes,
        "wrapper_sha256": hashlib.sha256(Path(resume.__file__).read_bytes()).hexdigest(),
        "coordinator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sequential_parent_pids_at_start": parents,
        "initial": initial,
        "locking": "Intentionally bypasses parent global mutex; "
        "every document is guarded by the same nonblocking flock.",
        "expected_parent_exit": "Parent finishes its current locked document. "
        "On the next claimed or completed document, "
        "its existing lock or changed-record guard aborts before a new API call. "
        "That coordinator handoff exit is expected, not an inference failure.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    resume.runner.atomic_json(report_path, coordination)
    if args.execute:
        load_dotenv(ROOT / ".env")
    results = []

    def worker(row):
        try:
            return process_document(row, freeze, args.execute)
        except Exception as exc:
            # Preserve document failure accounting inside continue_one; this
            # describes coordinator preflight failures without exposing bodies.
            return {"id": row["id"], "status": "coordinator_error", "error_type": type(exc).__name__}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(worker, inventory):
            results.append(result)
            if result["status"] not in ("ineligible_or_already_finished",):
                print(json.dumps(result), flush=True)
    final = outcome_snapshot(inventory)
    coordination.update(
        completed_at_utc=datetime.now(UTC).isoformat(),
        results=results,
        final=final,
        sequential_parent_pids_at_end=active_sequential_parents(),
    )
    resume.runner.atomic_json(report_path, coordination)
    print(json.dumps({"coordination_artifact": str(report_path), "outcomes": final["outcome_count"]}), flush=True)
    if args.execute and (
        final["outcome_count"] != args.expected_outcomes
        or final["immutable_original_count"] != args.expected_outcomes
        or final["unfinished_ids"]
        or final["unclaimed_or_parent_running_ids"]
    ):
        parser.error(
            "Some outcomes are still pending (possibly the parent's locked document); run --verify-only after it exits"
        )


if __name__ == "__main__":
    main()
