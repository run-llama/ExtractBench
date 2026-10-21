"""Generate Harbor tasks alongside the existing Hugging Face JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from extract_bench.data.download import _write_sidecar, dataset_config
from extract_bench.evaluation.score_case import load_extract_test_case

PAPER_URL = "https://arxiv.org/html/2607.29677v1#A3.SS1"
# Appendix C.1's shared Claude Code/Codex prompt. Only the document filename
# is standardized; the schema is inserted exactly as in the published prompt.
INSTRUCTION = (
    """Extract structured data from the document file `./document.pdf` in the current directory.

Use only local file inspection and shell commands. Do not use web search, network calls, browser tools, """
    """or external services. Temporary scratch files inside the current directory are OK.

Return a single JSON object conforming to this schema as your final answer:

{schema}

Rules:

- Use null for fields not present in the document.
- For list/array fields, enumerate every relevant row visible in the document; never collapse rows.
- For large regular tables, prefer writing and running a local script to parse/enumerate rows.
- For forms, prefer direct field extraction from the document content.
- Write the resulting JSON object to ./output.json and validate that it is valid JSON before stopping.
- Do not print the JSON to your assistant output.
"""
)

AGENT_DOCKERFILE = """FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \\
    poppler-utils tesseract-ocr curl git nodejs npm \\
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pymupdf==1.26.6 pypdf==6.4.0 pillow==12.0.0
WORKDIR /app
COPY document.pdf schema.json /app/
"""


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _task_id(case_id: str) -> str:
    if Path(case_id).is_absolute() or ".." in Path(case_id).parts:
        raise ValueError(f"Unsafe case ID: {case_id!r}")
    slug = re.sub(r"[^a-z0-9-]", "-", case_id.lower().replace("/", "--"))[:64].strip("-")
    return f"{slug}-{hashlib.sha256(case_id.encode()).hexdigest()[:10]}"


def _task_config(row: Mapping[str, Any], repo: str, revision: str, scorer_revision: str, timeout: int) -> str:
    return f"""schema_version = "1.3"
artifacts = ["/app/output.json"]

[metadata]
source_repo = {json.dumps(repo)}
source_revision = {json.dumps(revision)}
source_case_id = {json.dumps(row["id"])}
source_split = {json.dumps(row["category"])}
scorer_revision = {json.dumps(scorer_revision)}
prompt_source = "{PAPER_URL}"
tags = {json.dumps(row.get("tags", []))}

[agent]
timeout_sec = {timeout}

[environment]
cpus = 2
memory_mb = 4096
storage_mb = 10240

[verifier]
environment_mode = "separate"
timeout_sec = 1800

[verifier.environment]
cpus = 2
memory_mb = 8192
"""


def export_snapshot(snapshot: Path, output: Path, dataset: str, revision: str, scorer_revision: str) -> dict[str, Any]:
    """Export a pinned HF snapshot; reruns repair partial tasks deterministically.

    ``output`` is a staging directory for additions to the same HF repository.
    Existing non-generated data is never deleted or rewritten. A source-case
    removal is refused so stale tasks cannot silently survive a refresh.
    """
    for name, value in (("revision", revision), ("scorer_revision", scorer_revision)):
        if re.fullmatch(r"[0-9a-f]{40}", value) is None:
            raise ValueError(f"{name} must be a full commit SHA")
    config = dataset_config(dataset)
    rows: list[dict[str, Any]] = []
    task_ids: set[str] = set()
    for split in config.splits:
        for line in (snapshot / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            if len(line.strip()) == 0:
                continue
            row = json.loads(line)
            case_id = row["id"]
            if row["category"] != split or not case_id.startswith(split + "/"):
                raise ValueError(f"Case {case_id!r} does not belong to {split}")
            # A flat task directory supports Harbor's default tasks/ discovery.
            task_id = _task_id(case_id)
            if task_id in task_ids:
                raise ValueError(f"Colliding case ID: {case_id!r}")
            task_ids.add(task_id)
            rows.append(row)
    tasks = output / "tasks"
    if tasks.exists():
        stale = {path.name for path in tasks.iterdir()} - task_ids
        if len(stale) > 0:
            raise ValueError(f"Output contains stale tasks; use a new staging directory: {sorted(stale)}")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "harbor-manifest.json"
    manifest_path.unlink(missing_ok=True)
    manifest: dict[str, Any] = {
        "source_repo": config.repo,
        "source_revision": revision,
        "scorer_revision": scorer_revision,
        "prompt_source": PAPER_URL,
        "reward": "extract_unified_value_f1",
        "agent_timeout_sec": 1800 if dataset == "pro" else 1200,
        "source_files": {},
        "tasks": {},
    }
    # Retain the ordinary datasets loading contract in the staged repo too.
    for split in config.splits:
        source = snapshot / f"{split}.jsonl"
        target = output / source.name
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
        manifest["source_files"][source.name] = _sha256(source)
    card = (snapshot / "README.md").read_text(encoding="utf-8")
    if "\n- harbor\n" not in card:
        card = card.replace("tags:\n", "tags:\n- rl-environment\n- harbor\n", 1)
    card += f"""

## Harbor

Requires Harbor 0.24.0 or later, Git with Git LFS, and Docker.
Run `git lfs install` before the first use, then run the tasks directly from this repository:

```bash
harbor run --repo https://huggingface.co/datasets/{config.repo} -p tasks -a codex -m MODEL
```

Instructions reproduce the [published coding-agent prompt]({PAPER_URL}), with a standardized document filename.
Agents receive only `document.pdf` and `schema.json` and write `output.json`.
The separate verifier uses the existing ExtractBench evaluator at commit `{scorer_revision}`.
The reward is Value F1 in [0, 1]; `metrics.json` also records perfect-document status.
This prompt does not request citations, so this lane does not measure grounding.
The agent timeout is {manifest["agent_timeout_sec"]} seconds per document.

The JSONL files and ordinary Hugging Face dataset configuration are preserved.
`harbor-manifest.json` records source revisions and hashes for reproducibility.
For a no-inference packaging smoke test, use `-a oracle -l 1` instead of `-a codex -m MODEL`.
The oracle copies the reference output; it checks packaging and grading, not extraction ability.
"""
    (output / "README.md").write_text(card, encoding="utf-8")
    attributes = (snapshot / ".gitattributes").read_text(encoding="utf-8")
    pdf_rule = "tasks/**/document.pdf filter=lfs diff=lfs merge=lfs -text"
    if pdf_rule not in attributes.splitlines():
        attributes = attributes.rstrip() + "\n" + pdf_rule + "\n"
    (output / ".gitattributes").write_text(attributes, encoding="utf-8")
    for index, row in enumerate(rows, start=1):
        task_id = _task_id(row["id"])
        task = tasks / task_id
        env = task / "environment"
        tests = task / "tests"
        solution = task / "solution"
        for directory in (env, tests, solution):
            directory.mkdir(parents=True, exist_ok=True)
        # Reuse HF reconstruction so field rules and row-identity metadata
        # reach the same loader as an ordinary `extract-bench download`.
        reference = _write_sidecar(row, tests, snapshot, force=True)
        source_pdf = reference.with_name(reference.name.removesuffix(".test.json") + ".pdf")
        shutil.move(str(source_pdf), env / "document.pdf")
        reference = reference.rename(tests / "reference.test.json")
        _write_json(env / "schema.json", json.loads(row["data_schema"]))
        case = load_extract_test_case(reference, source_file=env / "document.pdf")
        _write_json(solution / "expected.json", case.expected_output)
        (task / "instruction.md").write_text(INSTRUCTION.format(schema=row["data_schema"]), encoding="utf-8")
        (task / "task.toml").write_text(
            _task_config(row, config.repo, revision, scorer_revision, manifest["agent_timeout_sec"]), encoding="utf-8"
        )
        (env / "Dockerfile").write_text(AGENT_DOCKERFILE, encoding="utf-8")
        # Pin the public evaluator, but ship this adapter with the task so
        # exports also work before the adapter has landed on upstream main.
        (tests / "Dockerfile").write_text(
            "FROM python:3.12-slim-bookworm\n"
            "RUN apt-get update && apt-get install -y --no-install-recommends git "
            "&& rm -rf /var/lib/apt/lists/*\n"
            f'RUN pip install --no-cache-dir "llama-extract-bench @ '
            f'git+https://github.com/run-llama/ExtractBench.git@{scorer_revision}" "parse-bench==1.0.4"\n'
            "WORKDIR /app\nCOPY . /tests/\n",
            encoding="utf-8",
        )
        shutil.copyfile(Path(__file__).with_name("verifier.py"), tests / "verifier.py")
        (tests / "test.sh").write_text("#!/bin/bash\nset -euo pipefail\npython /tests/verifier.py\n", encoding="utf-8")
        (solution / "solve.sh").write_text(
            "#!/bin/bash\nset -euo pipefail\ncp /solution/expected.json /app/output.json\n", encoding="utf-8"
        )
        for script in (tests / "test.sh", solution / "solve.sh"):
            script.chmod(0o755)
        manifest["tasks"][task_id] = {
            "case_id": row["id"],
            "pdf": row["pdf"],
            "files": {str(path.relative_to(task)): _sha256(path) for path in sorted(task.rglob("*")) if path.is_file()},
        }
        print(f"[{index}/{len(rows)}] exported {row['id']}", flush=True)
    _write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("extractbench", "pro"), required=True)
    parser.add_argument("--revision", required=True, help="Full Hugging Face dataset commit SHA")
    parser.add_argument("--scorer-revision", required=True, help="Full public ExtractBench commit SHA")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--snapshot", type=Path, help="Already downloaded HF snapshot (otherwise download pinned source)"
    )
    args = parser.parse_args()
    snapshot = args.snapshot
    if snapshot is None:
        from huggingface_hub import snapshot_download

        config = dataset_config(args.dataset)
        snapshot = Path(
            snapshot_download(
                config.repo,
                repo_type="dataset",
                revision=args.revision,
                allow_patterns=["README.md", ".gitattributes", "*.jsonl", "docs/**"],
            )
        )
    export_snapshot(snapshot, args.output, args.dataset, args.revision, args.scorer_revision)


if __name__ == "__main__":
    main()
