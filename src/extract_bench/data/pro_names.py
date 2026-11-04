"""Translate the original Pro layout to the canonical challenge names.

Only identity metadata is translated. Schemas, expected/extracted values,
model responses, and scoring payloads must never undergo text replacement.
Harbor exports use canonical IDs directly and do not use this local migrator.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

LEGACY_PRO_SPLITS: Mapping[str, str] = {
    "document_conventions": "interpretation_rules",
    "house_vocabulary": "custom_terminology",
    "house_counting_rules": "conditional_aggregation",
    "drawn_relations": "relational_diagrams",
    "charts_inside_documents": "charts_and_maps",
    "long_chart_sheets": "long_single_pages",
}
PRO_SPLITS = (
    "interpretation_rules",
    "custom_terminology",
    "conditional_aggregation",
    "relational_diagrams",
    "charts_and_maps",
    "long_single_pages",
)


def canonical_pro_split(value: str) -> str:
    return LEGACY_PRO_SPLITS.get(value, value)


def canonical_pro_path(value: str) -> str:
    """Translate whole path components, retaining separators and document stems."""
    return "".join(canonical_pro_split(part) for part in re.split(r"([/\\])", value))


def canonical_pro_tag(value: str) -> str:
    prefix, separator, split = value.partition(":")
    return f"{prefix}:{canonical_pro_split(split)}" if separator == ":" and prefix == "challenge" else value


def translate_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a sidecar/result envelope without descending into document content."""
    translated = dict(payload)
    for key in ("test_id", "example_id", "group", "file_path", "source_file_path"):
        if isinstance(translated.get(key), str):
            translated[key] = canonical_pro_path(translated[key])
    if isinstance(translated.get("tags"), list):
        translated["tags"] = [canonical_pro_tag(tag) if isinstance(tag, str) else tag for tag in translated["tags"]]
    for key, fields in (("request", ("example_id", "source_file_path")), ("output", ("example_id",))):
        if isinstance(translated.get(key), dict):
            envelope = dict(translated[key])
            for field in fields:
                if isinstance(envelope.get(field), str):
                    envelope[field] = canonical_pro_path(envelope[field])
            translated[key] = envelope
    if isinstance(translated.get("per_split"), dict):
        counts = translated["per_split"]
        normalized = {canonical_pro_split(key): value for key, value in counts.items()}
        if len(normalized) != len(counts):
            raise ValueError("Manifest contains both original and canonical Pro split names")
        translated["per_split"] = normalized
    return translated


def migrate_pro_layout(root: Path, write: bool = False) -> tuple[str, ...]:
    """Plan, then optionally migrate one reconstructed dataset or pipeline root.

    Conflicting directories and malformed JSON fail before any writes. JSON
    envelopes are written atomically before directory renames; interrupted runs
    can be repeated. PDFs and model/answer content are never rewritten. Derived
    reports should be regenerated from the migrated predictions afterward.
    """
    if not root.is_dir():
        raise ValueError(f"Not a dataset or pipeline directory: {root}")
    moves = []
    for old, new in LEGACY_PRO_SPLITS.items():
        source, target = root / old, root / new
        if source.exists() or source.is_symlink():
            if source.is_symlink() or not source.is_dir() or target.exists():
                raise ValueError(f"Cannot migrate {source}: unsafe source or target {target} already exists")
            moves.append((source, target))
    changed = []
    suffixes = (".test.json", ".result.json", ".raw.json", ".evaluation.json", ".eval.json")
    for path in sorted(root.rglob("*.json")):
        if not path.name.endswith(suffixes) and path.name != ".extract_bench_manifest.json":
            continue
        if path.is_symlink():
            raise ValueError(f"Refusing to rewrite a symlink: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a JSON object: {path}")
        if translate_identity(payload) != payload:
            changed.append(path)
    plan = tuple(
        [f"update {path.relative_to(root)}" for path in changed] + [f"rename {a.name} -> {b.name}" for a, b in moves]
    )
    if write:
        for path in changed:
            payload = translate_identity(json.loads(path.read_text(encoding="utf-8")))
            temporary = path.with_name(path.name + ".pro-names-pending")
            temporary.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
            temporary.replace(path)
        for source, target in moves:
            source.rename(target)
    return plan


def repair_pro_layout(root: Path) -> None:
    """Migrate released split directories before data discovery or cache lookup.

    Canonical trees need only the directory checks. Legacy trees use the full
    migration preflight so malformed metadata and conflicting paths stop the
    command before inference can replace a cached prediction.
    """
    if not any((root / old).exists() or (root / old).is_symlink() for old in LEGACY_PRO_SPLITS):
        return
    changes = migrate_pro_layout(root, write=True)
    print(f"Updated original Pro names in {root} ({len(changes)} changes).")
