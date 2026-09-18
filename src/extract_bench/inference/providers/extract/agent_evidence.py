"""Evidence sidecar shared by coding-agent extract providers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from extract_bench.schemas.extract_output import FieldCitation


@dataclass
class EvidenceStats:
    cells: int = 0
    with_page: int = 0
    with_bbox: int = 0
    malformed_bbox: int = 0
    unwrapped_cells: int = 0
    cells_with_extra_keys: int = 0
    dropped_entries: int = 0
    envelope_missing_data: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "cells": self.cells,
            "with_page": self.with_page,
            "with_bbox": self.with_bbox,
            "malformed_bbox": self.malformed_bbox,
            "unwrapped_cells": self.unwrapped_cells,
            "cells_with_extra_keys": self.cells_with_extra_keys,
            "dropped_entries": self.dropped_entries,
            "envelope_missing_data": self.envelope_missing_data,
        }


def agent_evidence_instruction(citations_filename: str = "citations.json") -> str:
    return (
        f"- Also write ./{citations_filename}: a JSON array of evidence objects, "
        "one per non-null value you extracted, each with the keys "
        '"field_path", "page" and "bbox".\n'
        '  - "field_path" is the dotted path of the field in your output.json, '
        "with a bracketed index for array rows: `invoice_number`, "
        "`line_items[0].amount`.\n"
        '  - "page" is the 1-indexed page the value was read from.\n'
        '  - "bbox" is [x, y, width, height] as fractions of that page\'s '
        "width and height in [0, 1], origin at the TOP-LEFT corner of the "
        "page.\n"
        "  - Box the value itself -- the filled-in text, number, or checkbox "
        "mark -- not the surrounding row, label, or form section.\n"
        "  - You may use any tool available to locate values on the page "
        "(e.g. a PDF text-extraction library that reports word rectangles, or "
        "reading the rendered page yourself). Convert PDF-space rectangles to "
        "the top-left-origin fractions above before writing them.\n"
        "  - Omit a field entirely rather than guessing a location for it."
    )


def read_agent_citations_file(citations_path: Path) -> Any:
    if not citations_path.exists():
        return None
    try:
        return json.loads(citations_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def citations_from_agent_file(
    entries: Any, *, source: str = "agent_evidence"
) -> tuple[list[FieldCitation], EvidenceStats]:
    stats = EvidenceStats()
    citations: list[FieldCitation] = []
    if not isinstance(entries, list):
        return citations, stats

    for entry in entries:
        if not isinstance(entry, dict):
            stats.dropped_entries += 1
            continue
        field_path = entry.get("field_path") or entry.get("path")
        if not isinstance(field_path, str) or not field_path:
            stats.dropped_entries += 1
            continue
        stats.cells += 1
        page = _coerce_page(entry.get("page"))
        if page is None:
            continue
        stats.with_page += 1
        bbox, malformed = _normalize_bbox(entry.get("bbox"))
        if malformed:
            stats.malformed_bbox += 1
        if bbox is not None:
            stats.with_bbox += 1
        reference_text = entry.get("value") if isinstance(entry.get("value"), str) else None
        citations.append(
            FieldCitation(
                field_path=field_path,
                page=page,
                bbox=bbox,
                reference_text=reference_text,
                source=source,
            )
        )
    return citations, stats


def _normalize_bbox(raw: Any) -> tuple[list[float] | None, bool]:
    if raw is None:
        return None, False
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None, True
    try:
        x, y, width, height = (float(v) for v in raw[:4])
    except (TypeError, ValueError):
        return None, True
    if width <= 0 or height <= 0:
        return None, True
    if not (-0.02 <= x <= 1.02 and -0.02 <= y <= 1.02):
        return None, True
    if width > 1.02 or height > 1.02:
        return None, True

    x = min(max(x, 0.0), 1.0)
    y = min(max(y, 0.0), 1.0)
    width = min(width, 1.0 - x)
    height = min(height, 1.0 - y)
    if width <= 0 or height <= 0:
        return None, True
    return [round(x, 6), round(y, 6), round(width, 6), round(height, 6)], False


def _coerce_page(raw: Any) -> int | None:
    if isinstance(raw, bool) or raw is None:
        return None
    try:
        page = int(raw)
    except (TypeError, ValueError):
        return None
    return page if page >= 1 else None
