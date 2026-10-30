"""Prepare an unpublished Harbor Hub package from a validated canonical HF export.

Requires Harbor 0.24.0. This command only writes local files. It adds package
names/schema 1.4, validates that execution settings are unchanged, and computes
content-pinned task references. It never publishes or creates legacy aliases.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from harbor.models.dataset.manifest import DatasetFileRef, DatasetInfo, DatasetManifest, DatasetTaskRef
from harbor.models.task.config import TaskConfig
from harbor.publisher.packager import Packager


def stage(export: Path, output: Path, revision: str) -> None:
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("HF revision must be a full commit SHA")
    export, output = export.resolve(), output.resolve()
    if export == output or export in output.parents or output in export.parents:
        raise ValueError("Export and Hub staging directories must be separate")
    manifest = json.loads((export / "harbor-manifest.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    tasks = []
    for task_id, entry in manifest["tasks"].items():
        source, target = export / "tasks" / task_id, output / "tasks" / task_id
        if target.exists():
            raise ValueError(f"Use a fresh Hub staging directory: {target}")
        for relative, expected in entry["files"].items():
            if Packager.compute_file_hash(source / relative) != expected:
                raise ValueError(f"Export changed after validation: {task_id}/{relative}")
        shutil.copytree(
            source,
            target,
            copy_function=lambda a, b: os.link(a, b) if Path(a).suffix == ".pdf" else shutil.copyfile(a, b),
        )
        config_path = target / "task.toml"
        old = TaskConfig.model_validate_toml(config_path.read_text())
        text = config_path.read_text().replace('schema_version = "1.3"', 'schema_version = "1.4"', 1)
        name = f"llamaindex/{task_id}"
        text += f'\n[task]\nname = "{name}"\nversion = "hf-{revision[:12]}"\n'
        current = TaskConfig.model_validate_toml(text)
        assert old.model_dump(exclude={"schema_version", "task"}) == current.model_dump(
            exclude={"schema_version", "task"}
        )
        config_path.write_text(text)
        digest, _ = Packager.compute_content_hash(target)
        tasks.append(DatasetTaskRef(name=name, digest=f"sha256:{digest}"))
    readme = (export / "README.md").read_text()
    readme += (
        "\n## Harbor Hub release\n\n"
        f"This candidate is based on HF commit `{revision}` and uses canonical task names. "
        "Original Harbor task names have no aliases in this release.\n\n"
        f"```bash\nharbor run -d llamaindex/extractbench-pro@hf-{revision[:12]} "
        "-a codex -m MODEL --allow-agent-host MODEL_API_HOST\n```\n"
    )
    (output / "README.md").write_text(readme)
    shutil.copyfile(Path(__file__).resolve().parents[1] / "LICENSE", output / "LICENSE")
    (output / "SOURCE.json").write_text(
        json.dumps(
            {
                "repo": manifest["source_repo"],
                "hf_revision": revision,
                "data_revision": manifest["source_revision"],
                "scorer_revision": manifest["scorer_revision"],
                "tasks": [task.model_dump() for task in tasks],
            },
            indent=2,
        )
        + "\n"
    )
    files = [
        DatasetFileRef(path=name, digest=f"sha256:{Packager.compute_file_hash(output / name)}")
        for name in ("README.md", "LICENSE", "SOURCE.json")
    ]
    dataset = DatasetManifest(
        dataset=DatasetInfo(
            name="llamaindex/extractbench-pro",
            version=f"hf-{revision[:12]}",
            description=(
                "ExtractBench Pro: 100 synthetic document extraction tasks across six challenges; "
                "deterministic Value F1 reward."
            ),
        ),
        tasks=tasks,
        files=files,
    )
    (output / "dataset.toml").write_text(dataset.to_toml())
    print(f"Staged {len(tasks)} canonical task packages and dataset.toml; nothing published")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hf-revision", required=True)
    args = parser.parse_args()
    stage(args.export, args.output, args.hf_revision)
