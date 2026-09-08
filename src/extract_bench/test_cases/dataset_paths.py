"""Helpers for resolving benchmark dataset paths and names."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def dataset_name_from_test_cases_dir(test_cases_dir: str) -> str:
    """Return dataset path relative to the data root (e.g. extract/short)."""
    normalized = str(test_cases_dir or "").strip()
    if not normalized:
        return "unknown"
    path_parts = Path(normalized).parts
    if "data" in path_parts:
        data_index = path_parts.index("data")
        if data_index + 1 < len(path_parts):
            return "/".join(path_parts[data_index + 1 :])
    return Path(normalized).name


def dataset_name_from_metadata(metadata: dict[str, Any]) -> str:
    """Extract dataset name from inference or benchmark metadata."""
    return dataset_name_from_test_cases_dir(str(metadata.get("test_cases_dir", "")))


# A version directory: ``v0.2``, ``v1``, ``v0.3-rc`` ...
VERSION_SEGMENT_PATTERN = re.compile(r"^v[\w.-]+$", re.IGNORECASE)


def is_version_segment(segment: str) -> bool:
    return bool(VERSION_SEGMENT_PATTERN.match(str(segment or "").strip()))


def parse_dataset_name(dataset_name: str) -> tuple[str, str | None]:
    """Split a dataset name into base and version.

    Datasets live under the data root as ``<base>/<version>`` where ``<base>`` may itself be
    nested inside a collection folder: ``extract/v0.2`` but also ``extract/short/v0.2`` (base
    ``extract/short``) and ``extract/long/v0.3/qa`` (base ``extract/long``, version ``v0.3/qa``).
    The version starts at the LAST segment that looks like a version directory; splitting at the
    first slash instead would file every ``extract/*`` dataset under one family. Names without any
    version-like segment keep the legacy first-slash split.
    """

    normalized = str(dataset_name or "").strip().strip("/")
    if not normalized or normalized == "unknown":
        return normalized, None
    parts = normalized.split("/")
    if len(parts) == 1:
        return normalized, None
    for index in range(len(parts) - 1, 0, -1):
        if is_version_segment(parts[index]):
            return "/".join(parts[:index]), "/".join(parts[index:])
    return parts[0], "/".join(parts[1:])


def join_dataset_name(base: str, version: str | None) -> str:
    """Join dataset base and version into a single dataset name."""
    base_name = str(base or "").strip().strip("/")
    version_name = str(version or "").strip().strip("/")
    if not base_name:
        return version_name
    if not version_name:
        return base_name
    return f"{base_name}/{version_name}"


def resolve_dataset_dir(data_root: Path, dataset_name: str) -> Path:
    """Resolve and validate a dataset directory under the bench data root."""
    normalized = str(dataset_name or "").strip().strip("/")
    if not normalized:
        raise ValueError("dataset_name is required")
    dataset_path = (data_root / normalized).resolve()
    data_root_resolved = data_root.resolve()
    try:
        dataset_path.relative_to(data_root_resolved)
    except ValueError as exc:
        raise ValueError(f"Invalid dataset_name: {dataset_name!r}") from exc
    return dataset_path
