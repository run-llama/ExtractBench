"""Exercise real public cases through Harbor without making model calls.

The smoke snapshots contain one original row each, with empty remaining splits.
They are test fixtures, not publishable full-dataset exports.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import hf_hub_download

from extract_bench.data.download import dataset_config
from extract_bench.harbor.export import export_snapshot

SCORER = "2f27f100730afc7c395602e26a2a87e07ea66b31"
CASES = (
    ("extractbench", "51bf3a03eb7feeaef8952be1fc2fc7f924ee72dc", "short/dd1155_schedule_continuation_0002"),
    ("pro", "16f6672e06cf77491ec6d18c817d67864b90d845", "house_counting_rules/vendor_statement_reconciliation_01"),
)


def smoke(output: Path) -> None:
    harbor = shutil.which("harbor")
    if harbor is None:
        raise RuntimeError("Install Harbor 0.24.0 or later before running the container smoke tests")
    summaries = []
    for dataset, revision, case_id in CASES:
        config = dataset_config(dataset)
        snapshot = output / "fixtures" / dataset
        snapshot.mkdir(parents=True, exist_ok=True)

        def download(filename: str, repo: str = config.repo, ref: str = revision) -> Path:
            return Path(hf_hub_download(repo, filename, repo_type="dataset", revision=ref))

        selected_split = case_id.split("/", 1)[0]
        rows = [
            json.loads(line)
            for line in download(f"{selected_split}.jsonl").read_text().splitlines()
            if len(line.strip()) > 0
        ]
        row = next(row for row in rows if row["id"] == case_id)
        for split in config.splits:
            text = json.dumps(row) + "\n" if split == selected_split else ""
            (snapshot / f"{split}.jsonl").write_text(text, encoding="utf-8")
        for filename in ("README.md", ".gitattributes", row["pdf"]):
            target = snapshot / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(download(filename), target)
        exported = output / "fixtures-exported" / dataset
        manifest = export_snapshot(snapshot, exported, dataset, revision, SCORER)
        task_id = next(iter(manifest["tasks"]))
        for agent, expected in (("oracle", 1.0), ("nop", 0.0)):
            name = f"{dataset}-{agent}"
            print(datetime.now(UTC).isoformat(), name, "START", flush=True)
            subprocess.run(
                [
                    harbor,
                    "run",
                    "-p",
                    str(exported / "tasks" / task_id),
                    "-a",
                    agent,
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
            if verdict["exception_info"] is not None or rewards != {"reward": expected}:
                raise RuntimeError(f"{name}: expected reward {expected}, got {verdict}")
            feedback = json.loads((trial.parent / "verifier/feedback.json").read_text())
            if agent == "oracle":
                if len(feedback["cells"]) == 0 or not math.isclose(
                    sum(cell["f1_credit"] for cell in feedback["cells"]), expected
                ):
                    raise RuntimeError(f"{name}: cell credits do not explain the reward")
            elif feedback["status"] != "invalid_submission" or len(feedback["cells"]) != 0:
                raise RuntimeError(f"{name}: missing output has misleading feedback")
            summaries.append({"dataset": dataset, "case_id": case_id, "agent": agent, "reward": expected})
            print(datetime.now(UTC).isoformat(), name, "COMPLETE", rewards, flush=True)
    (output / "container-smokes.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/harbor-smoke"))
    smoke(parser.parse_args().output.resolve())
