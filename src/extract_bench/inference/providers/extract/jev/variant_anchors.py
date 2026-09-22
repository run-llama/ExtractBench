"""Schema-guided record anchors group wrapped source rows before extraction."""

from __future__ import annotations

import re

from . import variant_localized, variant_rowrepair
from .variant_multipage import _leaves
from .variant_tables import _index, _question, _schema, _type

# Generic source-token families, independent of document domain or benchmark.
_FAMILIES = {
    "nine-digit identifier": re.compile(r"\d{9}"),
    "long numeric identifier": re.compile(r"\d{6,16}"),
    "mixed alphanumeric identifier": re.compile(r"(?=.*\d)(?=.*[A-Za-z])[A-Za-z\d-]{5,20}"),
    "date": re.compile(r"\d{1,4}[/-]\d{1,2}[/-]\d{1,4}"),
    "currency or decimal amount": re.compile(r"\(?[-+]?[$€£¥]?\d[\d,]*\.\d{1,4}\)?"),
    "integer or comma-separated count": re.compile(r"[-+]?\d[\d,]*"),
}


def anchor_candidates(rows):
    candidates = []
    maximum = max((len(row.cells) for row in rows), default=0)
    for col in range(maximum):
        for family, pattern in _FAMILIES.items():
            hits = [i for i, row in enumerate(rows) if col < len(row.cells) and pattern.fullmatch(row.cells[col])]
            if len(hits) < 2:
                continue
            # An ordinal column that matches nearly all one-cell prose lines is
            # not a useful boundary; families deliberately exclude arbitrary text.
            candidates.append(
                {"column": col, "family": family, "hits": hits, "examples": [rows[i].cells[col] for i in hits[:5]]}
            )
    # Prioritize recurrent layouts when a very wide report has >254 candidates.
    return sorted(candidates, key=lambda c: -len(c["hits"]))[:254]


def group_rows(rows, anchor):
    """Merge only sparse, nearby, aligned text continuations; keep all other rows."""
    if anchor is None:
        return [row.raw for row in rows], 0
    starts = set(anchor["hits"])
    output = []
    current = None
    last_line = None
    merged = 0
    for i, row in enumerate(rows):
        if i in starts:
            current = {"cells": list(row.cells), "starts": row.starts, "line": row.line}
            output.append(current)
            last_line = row.line
            continue
        # A continuation must be short text positioned under existing text cells.
        can_merge = current is not None and last_line is not None and row.line - last_line <= 2
        can_merge = can_merge and len(row.cells) <= max(2, len(current["cells"]) // 2)
        assignments = []
        if can_merge:
            for text, x in zip(row.cells, row.starts, strict=False):
                if not text or any(char.isdigit() for char in text):
                    can_merge = False
                    break
                nearest = min(range(len(current["starts"])), key=lambda c: abs(current["starts"][c] - x))
                prior = current["cells"][nearest]
                if abs(current["starts"][nearest] - x) > 4 or not re.search(r"[A-Za-z]", prior):
                    can_merge = False
                    break
                assignments.append((nearest, text))
        if can_merge:
            for col, text in assignments:
                current["cells"][col] += " " + text
            last_line = row.line
            merged += 1
        else:
            output.append(row.raw)
            current = None
            last_line = None
    return [(" | ".join(row["cells"]) if isinstance(row, dict) else row) for row in output], merged


class MappingCache:
    """Reuse mappings only when explicit nonnumeric pipe headers agree."""

    def __init__(self, client, pages):
        self.client, self.cache = client, {}
        self.headers = set()
        for page in pages:
            for line in page["text"].splitlines():
                if "|" in line and re.search(r"[A-Za-z]", line) and not re.search(r"\d", line):
                    self.headers.add(line.strip())

    def decide(self, state, questions):
        key = None
        if questions and all(re.fullmatch(r"f\d+", q) for q in questions):
            # Actual printed header membership is mandatory. Same-width tables
            # with differing labels never share mappings.
            prefix = str(state).split("Sample complete rows:", 1)[0]
            headers = tuple(sorted(h for h in self.headers if h in prefix))
            if headers:
                key = (headers, tuple((q["instructions"], len(q["criteria"])) for q in questions.values()))
                if key in self.cache:
                    return self.cache[key]
        answers = self.client.decide(state, questions)
        if key is not None:
            self.cache[key] = answers
        return answers


def extract(document, schema, client):
    initial = variant_localized.extract(document, variant_rowrepair._scalar_schema(schema, schema), client)
    holder = {"root": initial["data"]}
    evidence = initial.get("evidence", [])
    pages = document.get("pages") or [
        {"page_index": 0, "text": document.get("text", ""), "markdown": document.get("markdown", "")}
    ]
    prepared_pages = [{**p, "text": p.get("markdown") or p.get("text", "")} for p in pages]
    raw_pages = variant_rowrepair.source_rows({"pages": prepared_pages})
    diagnostics = []

    def walk(node, path, parent, key):
        node = _schema(node, schema)
        if _type(node) == "object":
            for name, child in node.get("properties", {}).items():
                walk(child, f"{path}.{name}".strip("."), parent[key], name)
            return
        if _type(node) != "array":
            return
        fields = ", ".join(".".join(names) for names, _ in _leaves(node.get("items", {}), schema))[:1500]
        transformed = []
        for page, rows in zip(pages, raw_pages, strict=True):
            markdown = page.get("markdown", "")
            has_markdown_table = bool(re.search(r"(?m)^\s*\|?\s*:?-{3,}:?\s*\|", markdown))
            candidates = [] if has_markdown_table else anchor_candidates(rows)
            anchor = None
            if candidates:
                # Samples identify structure only. Every matching source row remains
                # eligible, including records beyond these sample windows.
                representative = []
                for candidate in candidates[:8]:
                    for i in candidate["hits"][:2]:
                        representative.extend(rows[j].raw for j in range(max(0, i - 1), min(len(rows), i + 2)))
                state = "\n".join(dict.fromkeys(representative))[:9500]
                options = [
                    f"Column {c['column']}, {c['family']}; {len(c['hits'])} rows; examples: " + ", ".join(c["examples"])
                    for c in candidates
                ]
                answer = client.decide(
                    state,
                    {
                        "anchor": _question(
                            f"Which repeated column reliably marks the beginning of each {path} record? Fields: {fields}. Select absent if these are unrelated records.",  # noqa: E501
                            options,
                        )
                    },
                )
                selected = _index(answer.get("anchor", {}), len(candidates))
                if selected is not None:
                    anchor = candidates[selected]
            lines, merged = group_rows(rows, anchor)
            transformed.append({"page_index": page.get("page_index", 0), "text": "\n".join(lines)})
            diagnostics.append(
                {
                    "array": path,
                    "page_index": page.get("page_index", 0),
                    "anchor": {k: anchor[k] for k in ("column", "family")} if anchor else None,
                    "merged_continuations": merged,
                }
            )
        array_document = {"pages": transformed, "text": "\n\n".join(p["text"] for p in transformed)}
        array_key = path.split(".")[-1] or "records"
        array_schema = {
            "type": "object",
            "$defs": schema.get("$defs", {}),
            "definitions": schema.get("definitions", {}),
            "properties": {array_key: node},
        }
        result = variant_rowrepair.extract(array_document, array_schema, MappingCache(client, transformed))
        parent[key] = result["data"][array_key]
        for item in result.get("evidence", []):
            item = dict(item)
            if item.get("path", "").startswith(array_key + "["):
                item["path"] = path + item["path"][len(array_key) :]
            # Grouped rows are synthetic contexts; retain literal source cell text
            # and page attribution instead of pretending they have original geometry.
            item.pop("context", None)
            item.pop("line_index", None)
            evidence.append(item)

    walk(schema, "", holder, "root")
    return {"data": holder["root"], "evidence": evidence, "anchor_diagnostics": diagnostics}
