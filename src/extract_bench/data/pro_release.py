"""Stage the canonical Pro dataset layout without altering extraction content.

Run against a pinned HF snapshot. Publishing the staged files is a separate
review step; this script never writes to HF or Harbor Hub. The migration map
records every source/canonical case ID and the unchanged PDF SHA-256 digest.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from extract_bench.data.pro_names import LEGACY_PRO_SPLITS, canonical_pro_path, canonical_pro_split, canonical_pro_tag


def stage_pro_snapshot(source: Path, output: Path, revision: str) -> dict:
    source = source.resolve()
    output = output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and staging directories must be separate")
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Source revision must be a full commit SHA")
    output.mkdir(parents=True, exist_ok=True)
    migration = {"source_revision": revision, "split_names": dict(LEGACY_PRO_SPLITS), "cases": []}
    for old in LEGACY_PRO_SPLITS:
        new = canonical_pro_split(old)
        rows = []
        for line in (source / f"{old}.jsonl").read_text(encoding="utf-8").splitlines():
            if len(line.strip()) == 0:
                continue
            row = json.loads(line)
            if row["category"] != old or not row["id"].startswith(old + "/"):
                raise ValueError(f"Unexpected source identity: {row['id']}")
            original_id, original_pdf = row["id"], row["pdf"]
            pdf = source / original_pdf
            if not pdf.resolve().is_relative_to(source) or not pdf.is_file():
                raise ValueError(f"Missing or unsafe source PDF: {original_pdf}")
            row["id"] = canonical_pro_path(original_id)
            row["category"] = new
            row["pdf"] = canonical_pro_path(original_pdf)
            row["tags"] = [canonical_pro_tag(tag) for tag in row["tags"]]
            target = output / row["pdf"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(pdf, target)
            digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ValueError(f"PDF copy changed: {original_id}")
            migration["cases"].append(
                {
                    "old_id": original_id,
                    "id": row["id"],
                    "old_pdf": original_pdf,
                    "pdf": row["pdf"],
                    "pdf_sha256": digest,
                }
            )
            rows.append(json.dumps(row, ensure_ascii=False))
        (output / f"{new}.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    if len({case["id"] for case in migration["cases"]}) != len(migration["cases"]):
        raise ValueError("Renamed cases collide")
    for name in ("README.md", "eval.yaml", ".gitattributes"):
        text = (source / name).read_text(encoding="utf-8")
        for old, new in LEGACY_PRO_SPLITS.items():
            text = text.replace(old, new)
        if name == ".gitattributes":
            text = text.rstrip() + "\ndocs/**/*.pdf filter=lfs diff=lfs merge=lfs -text\n"
        (output / name).write_text(text, encoding="utf-8")
    path = output / "split-name-migration.json"
    path.write_text(json.dumps(migration, indent=2) + "\n", encoding="utf-8")
    return migration
