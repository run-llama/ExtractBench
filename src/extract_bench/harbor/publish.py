"""Open a Hugging Face dataset PR containing a validated Harbor export."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi

from extract_bench.data.download import dataset_config


def publish(output: Path) -> str:
    manifest = json.loads((output / "harbor-manifest.json").read_text(encoding="utf-8"))
    repo = manifest["source_repo"]
    datasets = {config.repo: config for config in (dataset_config("extractbench"), dataset_config("pro"))}
    if repo not in datasets:
        raise ValueError(f"Unexpected dataset repository: {repo!r}")
    sanitized_files = manifest.get("sanitized_files", {})
    split_filenames = {f"{split}.jsonl" for split in datasets[repo].splits}
    for filename, expected_hash in sanitized_files.items():
        if filename not in split_filenames or (output / filename).is_symlink():
            raise ValueError(f"Unexpected sanitized split file: {filename!r}")
        with (output / filename).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected_hash:
                raise ValueError(f"Sanitized split changed after generation: {filename}")
    expected_files: set[Path] = set()
    tasks = output / "tasks"
    for task_id, entry in manifest["tasks"].items():
        for relative, expected_hash in entry["files"].items():
            path = tasks / task_id / relative
            if not path.resolve().is_relative_to(tasks.resolve()) or path.is_symlink():
                raise ValueError(f"Unsafe task path: {path}")
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != expected_hash:
                raise ValueError(f"Task changed after generation: {path}")
            expected_files.add(path)
    if expected_files != {path for path in tasks.rglob("*") if path.is_file()}:
        raise ValueError("Task file set differs from manifest; regenerate the export")
    # The parent guard prevents applying this overlay to another source revision.
    commit = HfApi().upload_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=output,
        allow_patterns=["README.md", ".gitattributes", "harbor-manifest.json", "tasks/**", *sanitized_files],
        parent_commit=manifest["source_revision"],
        create_pr=True,
        commit_message="Refresh ExtractBench release artifacts",
    )
    if commit.pr_url is None:
        raise RuntimeError("Hugging Face did not return a dataset pull request URL")
    return commit.pr_url


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Generated export directory")
    args = parser.parse_args()
    print(publish(args.output))


if __name__ == "__main__":
    main()
