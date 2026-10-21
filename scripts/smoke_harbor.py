"""Validate current HF data and exercise Harbor with the current checkout.

Every case is checked without model calls. One case per dataset also crosses
its container boundary. Resolved revisions and wheel hashes allow reproducing
failures without freezing future checks to old code or data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import tomllib
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download

from extract_bench.data.download import dataset_config
from extract_bench.evaluation.score_case import load_extract_test_case
from extract_bench.harbor.export import export_snapshot
from extract_bench.harbor.verifier import score_prediction


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def progress(*items: Any) -> None:
    print(datetime.now(UTC).isoformat(), *items, flush=True)


def validate_source_splits(snapshot: Path, splits: Sequence[str]) -> None:
    expected = {f"{split}.jsonl" for split in splits}
    actual = {path.name for path in snapshot.glob("*.jsonl")}
    if actual != expected:
        raise ValueError(f"HF JSONL splits differ from the dataset configuration: {actual ^ expected}")


def validate_task(task: Path, row: Mapping[str, Any], source_pdf: Path) -> None:
    """Compare exported inputs/references directly to their source JSONL row."""
    expected = {
        "tags": row["tags"],
        "data_schema": json.loads(row["data_schema"]),
        "expected_output": json.loads(row["expected_output"]),
    }
    rules = json.loads(row["field_rules"])
    if len(rules) > 0:
        expected["_field_rules" if isinstance(rules, dict) else "test_rules"] = rules
    repeated = json.loads(row["repeated_structure"])
    if len(repeated) > 0:
        expected["_eval_row_identity"] = repeated
    reference = task / "tests/reference.test.json"
    if json.loads(reference.read_text()) != expected:
        raise ValueError(f"{row['id']}: grading reference differs from the HF row")
    if json.loads((task / "environment/schema.json").read_text()) != expected["data_schema"]:
        raise ValueError(f"{row['id']}: agent schema differs from the HF row")
    if digest(task / "environment/document.pdf") != digest(source_pdf):
        raise ValueError(f"{row['id']}: agent PDF differs from the HF document")
    config = tomllib.loads((task / "task.toml").read_text())
    if config["metadata"]["source_case_id"] != row["id"] or config["verifier"]["environment_mode"] != "separate":
        raise ValueError(f"{row['id']}: task configuration does not identify/isolate its case")
    if config["agent"].get("network_mode") != "allowlist" or config["agent"].get("allowed_hosts") != []:
        raise ValueError(f"{row['id']}: agent network policy permits external data access")
    case = load_extract_test_case(reference, source_file=source_pdf)
    if json.loads((task / "solution/expected.json").read_text()) != case.expected_output:
        raise ValueError(f"{row['id']}: oracle differs from the normal loader's reference")


def container_task(source: Path, target: Path, wheel: Path) -> None:
    """Use the built checkout in the smoke container, including uncommitted edits."""
    shutil.copytree(source, target, dirs_exist_ok=True)
    tests = target / "tests"
    shutil.copyfile(wheel, tests / wheel.name)
    config = tomllib.loads((source / "task.toml").read_text())
    requirement = json.dumps(
        "llama-extract-bench @ git+https://github.com/run-llama/ExtractBench.git@"
        + config["metadata"]["scorer_revision"]
    )
    dockerfile = (tests / "Dockerfile").read_text()
    if dockerfile.count(requirement) != 1:
        raise ValueError("Expected exactly one pinned scorer requirement in the production Dockerfile")
    # Copy only the package before pip install, preserving the shared dependency
    # layer across documents in production instead of copying references first.
    install = "RUN pip install --no-cache-dir " + requirement
    dockerfile = dockerfile.replace(install, f"COPY {wheel.name} /tmp/{wheel.name}\n" + install)
    (tests / "Dockerfile").write_text(dockerfile.replace(requirement, json.dumps(f"/tmp/{wheel.name}")))
    # Execute inside Harbor's agent phase, under the same policy as a coding agent.
    # An unauthenticated HEAD checks connectivity without making a model call.
    solve = target / "solution/solve.sh"
    solve.write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        "curl --noproxy '*' -sS -I --connect-timeout 10 --max-time 30 https://api.openai.com/ >/dev/null\n"
        "python - <<'PY'\n"
        "import socket, ssl\n"
        "tls = ssl._create_unverified_context()\n"
        "for host in ('huggingface.co', 'github.com', '1.1.1.1'):\n"
        "    try:\n"
        "        with socket.create_connection((host, 443), timeout=3) as connection:\n"
        "            with tls.wrap_socket(connection, server_hostname=host):\n"
        "                pass\n"
        "    except OSError:\n"
        "        continue\n"
        "    raise RuntimeError(f'Agent can reach forbidden destination: {host}')\n"
        "print('Model host reachable; dataset hosts and direct IP blocked', flush=True)\n"
        "PY\n" + solve.read_text()
    )


def validate_published(snapshot: Path, rows: Mapping[str, Any], source_files: Sequence[str]) -> None:
    """Detect published Harbor tasks left behind by an HF data update."""
    manifest = json.loads((snapshot / "harbor-manifest.json").read_text())
    if manifest["source_files"] != {name: digest(snapshot / name) for name in source_files}:
        raise ValueError("Published Harbor source hashes differ from the current HF JSONL")
    tasks = manifest["tasks"]
    if len(tasks) != len(rows) or {entry["case_id"] for entry in tasks.values()} != set(rows):
        raise ValueError("Published Harbor tasks do not cover the current HF rows")
    for task_id, entry in tasks.items():
        if Path(task_id).name != task_id or task_id in (".", ".."):
            raise ValueError(f"Unsafe published task ID: {task_id!r}")
        task = snapshot / "tasks" / task_id
        for relative, expected_hash in entry["files"].items():
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError(f"Unsafe published task file: {relative!r}")
            if digest(task / relative) != expected_hash:
                raise ValueError(f"{entry['case_id']}: published task hash differs for {relative}")
        row = rows[entry["case_id"]]
        validate_task(task, row, snapshot / row["pdf"])


def smoke(output: Path, extractbench_revision: str = "main", pro_revision: str = "main") -> None:
    import fitz

    harbor = shutil.which("harbor")
    if harbor is None:
        raise RuntimeError("Install Harbor 0.24.0 or later before running the container smoke tests")
    output.mkdir(parents=True, exist_ok=True)
    checkout = Path(__file__).resolve().parents[1]
    scorer_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(output / "wheel")], cwd=checkout, check=True)
    wheels = list((output / "wheel").glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("Expected one checkout wheel; use a fresh output directory")
    wheel = wheels[0]
    wheel_hash = digest(wheel)
    lock_hash = digest(checkout / "uv.lock")
    api = HfApi()
    data_summaries = []
    container_summaries = []
    for dataset, ref in (("extractbench", extractbench_revision), ("pro", pro_revision)):
        config = dataset_config(dataset)
        revision = api.repo_info(config.repo, repo_type="dataset", revision=ref).sha
        if revision is None:
            raise RuntimeError(f"{config.repo}: HF did not return a resolved commit")
        progress(dataset, "DOWNLOAD", revision, "checkout wheel", wheel_hash)
        snapshot = Path(
            snapshot_download(
                config.repo,
                repo_type="dataset",
                revision=revision,
                allow_patterns=[
                    "README.md",
                    ".gitattributes",
                    "*.jsonl",
                    "docs/**",
                    "harbor-manifest.json",
                    "tasks/**",
                ],
            )
        )
        validate_source_splits(snapshot, config.splits)
        exported = output / "exported" / dataset / revision
        manifest = export_snapshot(snapshot, exported, dataset, revision, scorer_revision)
        rows = {}
        for split in config.splits:
            if digest(snapshot / f"{split}.jsonl") != digest(exported / f"{split}.jsonl"):
                raise ValueError(f"{dataset}/{split}: source JSONL was changed by export")
            for line in (snapshot / f"{split}.jsonl").read_text().splitlines():
                if len(line.strip()) > 0:
                    row = json.loads(line)
                    rows[row["id"]] = row
        if len(rows) == 0 or len(rows) != len(manifest["tasks"]):
            raise ValueError(f"{dataset}: source/task coverage differs or is empty")
        published = (snapshot / "harbor-manifest.json").is_file()
        if published:
            validate_published(snapshot, rows, [f"{split}.jsonl" for split in config.splits])
        elif (snapshot / "tasks").exists():
            raise ValueError(f"{dataset}: published tasks have no manifest")
        case_summaries = []
        for index, (task_id, entry) in enumerate(manifest["tasks"].items(), start=1):
            task = exported / "tasks" / task_id
            row = rows[entry["case_id"]]
            progress(dataset, index, "/", len(rows), row["id"], "START")
            validate_task(task, row, snapshot / row["pdf"])
            with fitz.open(task / "environment/document.pdf") as pdf:
                if pdf.page_count == 0 or pdf.needs_pass:
                    raise ValueError(f"{row['id']}: PDF has no readable pages")
                pages = pdf.page_count
            # Exact wheel/reference hashes permit interrupted runs to resume,
            # while any data or scorer change forces regrading.
            key = hashlib.sha256(
                (
                    wheel_hash
                    + lock_hash
                    + entry["files"]["tests/reference.test.json"]
                    + entry["files"]["solution/expected.json"]
                ).encode()
            ).hexdigest()
            receipt = output / "audit" / dataset / task_id / f"{key}.json"
            cached = json.loads(receipt.read_text()) if receipt.exists() else None
            if cached is not None and cached["key"] == key:
                result = cached
            else:
                prediction = json.loads((task / "solution/expected.json").read_text())
                metrics, feedback = score_prediction(task / "tests/reference.test.json", prediction)
                reward = metrics["extract_unified_value_f1"]
                if (
                    reward != 1.0
                    or feedback["value_incomplete"]
                    or not math.isclose(sum(cell["f1_credit"] for cell in feedback["cells"]), reward)
                ):
                    raise ValueError(f"{row['id']}: reference reward/credit is inconsistent: {metrics}")
                result = {"key": key, "case_id": row["id"], "reward": reward, "counts": feedback["counts"]}
                receipt.parent.mkdir(parents=True, exist_ok=True)
                receipt.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            case_summaries.append({**result, "pages": pages})
            progress(dataset, row["id"], "COMPLETE", result["reward"])
        data_summaries.append(
            {
                "dataset": dataset,
                "source_revision": revision,
                "wheel_sha256": wheel_hash,
                "dependency_lock_sha256": lock_hash,
                "published_tasks_checked": published,
                "cases": case_summaries,
            }
        )
        (output / "data-validation.json").write_text(json.dumps(data_summaries, indent=2) + "\n", encoding="utf-8")
        # All cases were graded above; keep the container boundary check small.
        task_id = min(
            manifest["tasks"], key=lambda key: (exported / "tasks" / key / "environment/document.pdf").stat().st_size
        )
        task = output / "container-tasks" / dataset / revision / wheel_hash / task_id
        container_task(exported / "tasks" / task_id, task, wheel)
        for agent, expected_reward in (("oracle", 1.0), ("nop", 0.0)):
            name = f"{dataset}-{agent}-{revision[:8]}-{wheel_hash[:8]}-{datetime.now(UTC).strftime('%Y%m%d%H%M%S%f')}"
            progress(name, "START")
            subprocess.run(
                [
                    harbor,
                    "run",
                    "-p",
                    str(task),
                    "-a",
                    agent,
                    "--allow-agent-host",
                    "api.openai.com",
                    "-n",
                    "1",
                    "--jobs-dir",
                    str(output / "jobs"),
                    "--job-name",
                    name,
                ],
                check=True,
            )
            job = output / "jobs" / name
            result = json.loads((job / "result.json").read_text())
            if result["stats"]["n_errored_trials"] != 0:
                raise RuntimeError(f"{name} failed: {result['stats']}")
            trial = next(job.glob("*/result.json"))
            verdict = json.loads(trial.read_text())
            rewards = verdict["verifier_result"]["rewards"]
            if verdict["exception_info"] is not None or rewards != {"reward": expected_reward}:
                raise RuntimeError(f"{name}: expected reward {expected_reward}, got {verdict}")
            feedback = json.loads((trial.parent / "verifier/feedback.json").read_text())
            if agent == "oracle":
                if len(feedback["cells"]) == 0 or not math.isclose(
                    sum(cell["f1_credit"] for cell in feedback["cells"]), expected_reward
                ):
                    raise RuntimeError(f"{name}: cell credits do not explain the reward")
            elif feedback["status"] != "invalid_submission" or len(feedback["cells"]) != 0:
                raise RuntimeError(f"{name}: missing output has misleading feedback")
            container_summaries.append(
                {
                    "dataset": dataset,
                    "case_id": manifest["tasks"][task_id]["case_id"],
                    "agent": agent,
                    "reward": expected_reward,
                    "wheel_sha256": wheel_hash,
                }
            )
            progress(name, "COMPLETE", rewards)
        (output / "container-smokes.json").write_text(
            json.dumps(container_summaries, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/harbor-smoke"))
    parser.add_argument("--extractbench-revision", default="main")
    parser.add_argument("--pro-revision", default="main")
    args = parser.parse_args()
    smoke(args.output.resolve(), args.extractbench_revision, args.pro_revision)
