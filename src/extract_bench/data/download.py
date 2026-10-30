"""Download original ExtractBench or ExtractBench Pro from HuggingFace.

Datasets: https://huggingface.co/datasets/llamaindex/ExtractBench
          https://huggingface.co/datasets/llamaindex/ExtractBenchPro

The hub stores one JSONL row per (document, schema) test case plus the source
documents under ``docs/<split>/``. After download, each row is reconstructed
into the sidecar layout the test-case loader consumes natively:

    <local_dir>/
    ├── short/<stem>.pdf + <stem>.test.json
    ├── medium/<stem>.pdf + <stem>.test.json
    └── long/<stem>.pdf + <stem>.test.json

Row columns ``data_schema`` / ``expected_output`` / ``field_rules`` /
``repeated_structure`` are JSON-encoded strings (schemas differ per document).
``field_rules`` holds either the ``_field_rules`` dict shape or the
``test_rules`` list shape; it is written back to the matching sidecar key
based on its JSON type.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Set
from dataclasses import dataclass
from pathlib import Path

from extract_bench.data.pro_names import LEGACY_PRO_SPLITS, PRO_SPLITS, canonical_pro_tag, repair_pro_layout

DATASET_REPO = "llamaindex/ExtractBench"
DATASET_REPO_TYPE = "dataset"
TEST_DATA_REVISION = "test-data"

SPLITS = ["short", "medium", "long"]

# Default on-disk locations. Test data lives in a sibling subdirectory so the
# two datasets coexist and `--test` does not silently overlay or get masked
# by an existing full download.
DEFAULT_DATA_DIR = Path("./data")
DEFAULT_TEST_DATA_DIR = Path("./data/test")


@dataclass(frozen=True)
class Dataset:
    repo: str
    splits: tuple[str, ...]
    data_dir: Path


def dataset_config(dataset: str = "extractbench", test: bool = False) -> Dataset:
    if dataset == "extractbench":
        return Dataset(DATASET_REPO, tuple(SPLITS), DEFAULT_TEST_DATA_DIR if test else DEFAULT_DATA_DIR)
    if dataset == "pro":
        if test:
            raise ValueError("ExtractBench Pro has no --test subset. Use --group to run one challenge.")
        return Dataset(
            "llamaindex/ExtractBenchPro",
            PRO_SPLITS,
            Path("./data-pro"),
        )
    raise ValueError(f"Unknown dataset {dataset!r}. Choose 'extractbench' or 'pro'.")


# Records dataset ownership and the resolved HF revision. Per-split counts are
# filled only after reconstruction, so interrupted downloads remain not-ready.
MANIFEST_NAME = ".extract_bench_manifest.json"


def default_data_dir(test: bool = False, dataset: str = "extractbench") -> Path:
    """Return the default on-disk dataset path for the given mode.

    Used by every CLI surface that takes ``--test`` so the routing is
    consistent between ``download``, ``run`` and ``status``.
    """
    return dataset_config(dataset, test).data_dir


def read_manifest(data_dir: Path) -> dict | None:
    """Return the download manifest for ``data_dir``, or None if absent/invalid."""
    manifest_path = data_dir / MANIFEST_NAME
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return manifest if isinstance(manifest, dict) else None


def is_dataset_ready(data_dir: Path, dataset: str = "extractbench", test: bool = False) -> bool:
    """True when the manifest exists and every split has all recorded sidecars.

    An interrupted reconstruction leaves an incomplete manifest, so partial
    downloads report not-ready and get repaired by the next download.
    """
    config = dataset_config(dataset, test)
    if not data_dir.exists():
        return False
    manifest = read_manifest(data_dir)
    if manifest is None or manifest.get("repo") != config.repo or manifest.get("test") is not test:
        return False
    per_split = manifest.get("per_split")
    if not isinstance(per_split, dict):
        return False
    for split in config.splits:
        split_dir = data_dir / split
        expected = per_split.get(split)
        if type(expected) is not int or expected <= 0:
            return False
        sidecars = list(split_dir.rglob("*.test.json"))
        if len(sidecars) != expected:
            return False
        if any(not path.with_name(path.name.removesuffix(".test.json") + ".pdf").is_file() for path in sidecars):
            return False
    return True


def _remote_revision(revision: str | None, dataset: str = "extractbench") -> str | None:
    """Resolve ``revision`` to a commit sha, or None when the Hub is unreachable."""
    try:
        from huggingface_hub import HfApi

        return HfApi().repo_info(dataset_config(dataset).repo, repo_type=DATASET_REPO_TYPE, revision=revision).sha
    except Exception:
        return None


def _revision_is_stale(data_dir: Path, revision: str | None, dataset: str = "extractbench") -> bool:
    """True when the Hub has moved past the commit recorded in the manifest.

    Case counts alone cannot detect this: a branch can swap which documents it
    ships while keeping the same number per split. Offline, the answer is False
    so an existing download stays usable.
    """
    manifest = read_manifest(data_dir)
    local = (manifest or {}).get("revision")
    if not isinstance(local, str):
        return False
    remote = _remote_revision(revision, dataset)
    return remote is not None and remote != local


def _prune_orphans(data_dir: Path, keep: Set[Path], dataset: str = "extractbench") -> list[Path]:
    """Delete reconstructed sidecars (and their PDFs) no longer in the dataset."""
    removed = []
    for split in (*dataset_config(dataset).splits, *(LEGACY_PRO_SPLITS if dataset == "pro" else ())):
        split_dir = data_dir / split
        if not split_dir.is_dir():
            continue
        for sidecar in sorted(split_dir.rglob("*.test.json")):
            if sidecar in keep:
                continue
            pdf = sidecar.with_name(sidecar.name[: -len(".test.json")] + ".pdf")
            sidecar.unlink()
            removed.append(sidecar)
            if pdf.exists():
                pdf.unlink()
    return removed


def _write_sidecar(row: dict, split_dir: Path, snapshot_dir: Path, force: bool = False) -> Path:
    """Reconstruct one JSONL row into ``<stem>.pdf`` + ``<stem>.test.json``.

    Returns the sidecar path written.
    """
    pdf_rel = row["pdf"]  # e.g. docs/short/<stem>.pdf
    # The path comes from repo data — never let it escape the target dirs.
    pdf_rel_parts = Path(pdf_rel).parts
    if Path(pdf_rel).is_absolute() or ".." in pdf_rel_parts or not pdf_rel.startswith("docs/"):
        raise RuntimeError(f"Dataset row {row.get('id')} has an unsafe file path: {pdf_rel!r}")
    src_pdf = snapshot_dir / pdf_rel
    if not src_pdf.exists():
        raise FileNotFoundError(f"Dataset row {row.get('id')} references missing file: {pdf_rel}")

    stem = Path(pdf_rel).stem
    # Preserve any sub-group directories below docs/<split>/.
    rel_parent = Path(pdf_rel).parent.relative_to(Path("docs") / row["category"])
    dest_dir = split_dir / rel_parent
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_pdf = dest_dir / f"{stem}.pdf"
    if force or not dest_pdf.exists():
        shutil.copyfile(src_pdf, dest_pdf)

    sidecar: dict = {
        "tags": [canonical_pro_tag(tag) for tag in row.get("tags") or []],
        "data_schema": json.loads(row["data_schema"]),
        "expected_output": json.loads(row["expected_output"]),
    }
    rules = json.loads(row.get("field_rules") or "{}")
    if isinstance(rules, dict) and rules:
        sidecar["_field_rules"] = rules
    elif isinstance(rules, list) and rules:
        sidecar["test_rules"] = rules
    # The loader only reads `_eval_row_identity`; `_repeated_structure` is an
    # inert legacy key, so landing the column there would leave row identity
    # dormant for every consumer.
    repeated = json.loads(row.get("repeated_structure") or "{}")
    if repeated:
        sidecar["_eval_row_identity"] = repeated

    dest_sidecar = dest_dir / f"{stem}.test.json"
    dest_sidecar.write_text(json.dumps(sidecar, ensure_ascii=False), encoding="utf-8")
    return dest_sidecar


def download_dataset(
    data_dir: Path | None = None,
    force: bool = False,
    test: bool = False,
    dataset: str = "extractbench",
    revision: str | None = None,
) -> Path:
    """Download the ExtractBench dataset and reconstruct the sidecar layout.

    Uses huggingface_hub's snapshot_download (into the shared HF cache) to
    fetch the JSONL split files and source documents, then reconstructs the
    ``<split>/<stem>.pdf + <stem>.test.json`` layout under ``data_dir``.

    Args:
        data_dir: Local directory for the reconstructed dataset.
            Defaults to ./data (or ./data/test when ``test`` is set).
        force: Re-reconstruct (overwriting existing files) even if data
            already exists.
        test: Download the small original ExtractBench test dataset.
        dataset: "extractbench" (default) or "pro". Pro defaults to ./data-pro.

    Returns:
        Path to the reconstructed dataset directory.
    """
    from huggingface_hub import snapshot_download

    config = dataset_config(dataset, test)
    if data_dir is None:
        data_dir = config.data_dir

    # Two branches, same as ParseBench: the full dataset lives on the default
    # branch, the small smoke subset on `test-data`. The resolved commit is
    # recorded in the manifest so `status` can report exactly what is on disk.
    if test and revision is not None:
        raise ValueError("--test selects test-data; do not also pass --revision")
    if test:
        revision = TEST_DATA_REVISION

    existing_manifest = read_manifest(data_dir)
    if existing_manifest is not None and existing_manifest.get("repo") != config.repo:
        raise RuntimeError(
            f"{data_dir} belongs to {existing_manifest.get('repo')!r}, not {config.repo}. Use a different --data_dir."
        )
    if existing_manifest is not None and existing_manifest.get("test") != test:
        kind_existing = "test subset" if existing_manifest.get("test") else "full dataset"
        kind_requested = "test subset" if test else "full dataset"
        raise RuntimeError(
            f"{data_dir} already holds the {kind_existing}; refusing to overlay the "
            f"{kind_requested} into the same directory. Use a different --data_dir."
        )

    if dataset == "pro":
        repair_pro_layout(data_dir)

    if not force and is_dataset_ready(data_dir, dataset, test) and not _revision_is_stale(data_dir, revision, dataset):
        print(f"Dataset already downloaded at: {data_dir}")
        return data_dir

    label = "test dataset" if test else "dataset"
    print(f"Downloading {label} from HuggingFace: {config.repo}")
    if revision:
        print(f"Branch: {revision}")
    print(f"Destination: {data_dir}")

    snapshot_dir = Path(
        snapshot_download(
            repo_id=config.repo,
            repo_type=DATASET_REPO_TYPE,
            revision=revision,
            allow_patterns=[
                *[f"{split}.jsonl" for split in (*config.splits, *(LEGACY_PRO_SPLITS if dataset == "pro" else ()))],
                "docs/**",
            ],
        )
    )
    # huggingface_hub materializes snapshots under snapshots/<commit-sha>/.
    resolved_revision = snapshot_dir.name
    refresh = (
        force
        or not is_dataset_ready(data_dir, dataset, test)
        or (existing_manifest is not None and existing_manifest.get("revision") != resolved_revision)
    )
    # Keep dataset ownership through interruptions, but certify readiness only
    # after every split has been reconstructed from this snapshot.
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / MANIFEST_NAME).write_text(
        json.dumps({"repo": config.repo, "revision": resolved_revision, "test": test, "per_split": {}}),
        encoding="utf-8",
    )

    n_cases = 0
    per_split: dict[str, int] = {}
    written: set[Path] = set()
    for split in config.splits:
        source_names = [split] + [old for old, new in LEGACY_PRO_SPLITS.items() if dataset == "pro" and new == split]
        candidates = [
            snapshot_dir / f"{name}.jsonl" for name in source_names if (snapshot_dir / f"{name}.jsonl").is_file()
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"Expected exactly one source split for {split} at {snapshot_dir}, found {candidates}")
        jsonl_path = candidates[0]
        split_dir = data_dir / split
        split_cases = 0
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["category"] != jsonl_path.stem:
                raise ValueError(f"Row {row.get('id')} in {split}.jsonl has category {row['category']!r}")
            written.add(_write_sidecar(row, split_dir, snapshot_dir, force=refresh))
            split_cases += 1
        per_split[split] = split_cases
        n_cases += split_cases

    orphans = _prune_orphans(data_dir, written, dataset)
    if orphans:
        print(f"Removed {len(orphans)} document(s) no longer in the dataset:")
        for path in orphans:
            print(f"  - {path.relative_to(data_dir)}")

    # Complete counts are written last, after all sidecars and PDFs are in place.
    manifest = {
        "repo": config.repo,
        "revision": resolved_revision,
        "test": test,
        "cases": n_cases,
        "per_split": per_split,
    }
    (data_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    if not is_dataset_ready(data_dir, dataset, test):
        raise RuntimeError(f"Dataset download completed but validation failed. Check {data_dir} for missing files.")

    print(f"Dataset ready at: {data_dir} ({n_cases} test cases, revision {resolved_revision[:12]})")
    return data_dir
