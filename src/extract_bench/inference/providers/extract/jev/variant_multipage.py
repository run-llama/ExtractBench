"""Multi-page source-column extraction without a page or record count ceiling."""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

from .variant_tables import _candidates, _coerce, _index, _question, _schema, _type


@dataclass
class Region:
    page: int
    rows: list[list[str]]
    positions: list[float]

    def preview(self):
        # Samples describe a layout, never determine the retained record count.
        indices = sorted(set(range(min(5, len(self.rows)))) | {len(self.rows) // 2, len(self.rows) - 1})
        return "\n".join(f"row {i}: " + " | ".join(self.rows[i]) for i in indices if i >= 0)[:7000]


def _align(page, rows, tolerance):
    """Align cell starts, preserving missing interior cells instead of shifting."""
    clusters = []
    for _, cells in rows:
        for x, text in cells:
            if not text:
                continue
            nearest = min(clusters, key=lambda group: abs(statistics.median(group) - x), default=None)
            if nearest is not None and abs(statistics.median(nearest) - x) <= tolerance:
                nearest.append(x)
            else:
                clusters.append([x])
    # Headers can have different centering; recurring starts define columns.
    recurring = [g for g in clusters if len(g) >= 2]
    anchors = sorted(statistics.median(g) for g in (recurring if len(recurring) >= 2 else clusters))
    if len(anchors) < 2:
        return None
    output = []
    for _, cells in rows:
        aligned = [""] * len(anchors)
        for x, text in cells:
            col = min(range(len(anchors)), key=lambda i: abs(anchors[i] - x))
            aligned[col] = (aligned[col] + " " + text).strip()
        output.append(aligned)
    return Region(page, output, [y for y, _ in rows])


def regions(document):
    """Geometry first; fixed character positions when LiteParse has no boxes."""
    output = []
    pages = document.get("pages") or [{"page_index": 0, "text": document.get("text", "")}]
    for page in pages:
        page_id = page.get("page_index", 0)
        words = [w for w in page.get("words", []) if w.get("text", "").strip()]
        groups = []
        if words:
            height = statistics.median(max(1, w.get("height", 10)) for w in words)
            for word in sorted(words, key=lambda w: (w["y"], w["x"])):
                if not groups or abs(word["y"] - groups[-1][0]) > height * 0.55:
                    groups.append((word["y"], [word]))
                else:
                    groups[-1][1].append(word)
            rows = []
            for y, group in groups:
                cells = []
                right = None
                for word in sorted(group, key=lambda w: w["x"]):
                    x = word["x"]
                    if right is not None and x - right < height * 1.2:
                        start, value = cells[-1]
                        cells[-1] = (start, value + " " + word["text"])
                    else:
                        cells.append((x, word["text"]))
                    right = x + word.get("width", len(word["text"]) * height * 0.5)
                rows.append((y, cells))
            tolerance = max(3, page.get("width", 600) * 0.018)
            gap = height * 3.0
        else:
            rows = []
            for line_index, line in enumerate(page.get("text", "").splitlines()):
                if not line.strip() or re.fullmatch(r"[\s|:+\-=]+", line):
                    continue
                cells = [(m.start(), m.group().strip()) for m in re.finditer(r"\S(?:.*?\S)?(?= {2,}|\t|\||$)", line)]
                if "|" in line:
                    # Pipe offsets are unreliable for variable-width cells: use ordinal positions.
                    cells = [(i * 20, part.strip()) for i, part in enumerate(line.strip().strip("|").split("|"))]
                rows.append((line_index, cells))
            tolerance, gap = 2, 3
        block = []
        for row in rows:
            if block and row[0] - block[-1][0] > gap:
                region = _align(page_id, block, tolerance)
                if region:
                    output.append(region)
                block = []
            block.append(row)
        if block:
            region = _align(page_id, block, tolerance)
            if region:
                output.append(region)
    return output


def _leaves(node, root, path=()):
    node = _schema(node, root)
    if _type(node) == "object":
        for key, child in node.get("properties", {}).items():
            yield from _leaves(child, root, path + (key,))
    elif _type(node) != "array":
        yield path, node


def _assign(record, names, value):
    if not names:
        return value
    dest = record
    for name in names[:-1]:
        dest = dest.setdefault(name, {})
    dest[names[-1]] = value
    return record


def extract(document, schema, client):
    text = document.get("text") or "\n".join(p.get("text", "") for p in document.get("pages", []))
    layouts = regions(document)
    evidence = []

    def scalar(node, path):
        values = _candidates(text, path, node)
        # Candidate selection is field-local; never send the full document repeatedly.
        terms = re.findall(r"[a-z]{3,}", path.lower() + " " + node.get("description", "").lower())
        lines = text.splitlines()
        ranked = sorted(enumerate(lines), key=lambda pair: -sum(term in pair[1].lower() for term in terms))[:35]
        context_ids = sorted({j for i, _ in ranked for j in range(max(0, i - 1), min(len(lines), i + 2))})
        context = "\n".join(lines[i] for i in context_ids)[:9000]
        decision = client.decide(
            context,
            {
                "value": _question(
                    f"Choose the source value of {path}. Description: {node.get('description', '')}. Never choose the label.",  # noqa: E501
                    values,
                )
            },
        )
        idx = _index(decision.get("value", {}), len(values))
        if idx is None:
            return None
        evidence.append({"path": path, "source_text": values[idx]})
        return _coerce(values[idx], node)

    def array(node, path):
        fields = list(_leaves(node.get("items", {}), schema))
        if not fields:
            return []
        records = []
        field_description = "; ".join(
            ".".join(names) + ": " + field.get("description", "")[:220] for names, field in fields
        )[:6000]
        for region in layouts:
            preview = region.preview()
            relevant = client.decide(
                preview,
                {
                    "relevant": _question(
                        f"Does this table contain any records for array {path}? {node.get('description', '')[:1000]} Fields: {field_description}",  # noqa: E501
                        ["Yes, records belonging to this array occur here", "No, unrelated table"],
                    )
                },
            )
            if _index(relevant.get("relevant", {}), 2) != 0:
                continue
            width = len(region.rows[0])
            descriptions = [
                f"Column {c}: " + " / ".join(row[c] for row in region.rows[:8])[:1000] for c in range(width)
            ]
            maps = client.decide(
                preview,
                {
                    f"field{f}": _question(
                        f"Which source column provides {path}.{'.'.join(names)}? {field.get('description', '')[:1000]} Select absent when no source column provides this field.",  # noqa: E501
                        descriptions,
                    )
                    for f, (names, field) in enumerate(fields)
                },
            )
            columns = [_index(maps.get(f"field{f}", {}), width) for f in range(len(fields))]
            # Every source row is classified in bounded batches, including rows beyond
            # sample windows and every later page. There is no record-count cap.
            for start in range(0, len(region.rows), 20):
                batch = region.rows[start : start + 20]
                decisions = client.decide(
                    f"Array {path}. {node.get('description', '')[:800]}\nColumn samples:\n{preview}",
                    {
                        f"row{start + r}": _question(
                            f"Row {start + r}: " + " | ".join(row),
                            [
                                "A complete new record belonging to the requested array",
                                "Continuation of the preceding record, with wrapped text",
                                "Header, footer, subtotal, total, or unrelated text",
                            ],
                        )
                        for r, row in enumerate(batch)
                    },
                )
                for r, row in enumerate(batch):
                    kind = _index(decisions.get(f"row{start + r}", {}), 3)
                    if kind not in (0, 1):
                        continue
                    record = {} if _type(_schema(node.get("items", {}), schema)) == "object" else None
                    for f, (names, field) in enumerate(fields):
                        col = columns[f]
                        raw = row[col] if col is not None else None
                        value = _coerce(raw, field) if raw else None
                        record = _assign(record, names, value)
                        if raw:
                            evidence.append(
                                {
                                    "path": f"{path}[{max(0, len(records) - (kind == 1))}]." + ".".join(names),
                                    "source_text": raw,
                                    "page_index": region.page,
                                    "y": region.positions[start + r],
                                }
                            )
                    if kind == 1 and records and isinstance(record, dict):

                        def merge(old, new):
                            for k, value in new.items():
                                if isinstance(value, dict):
                                    merge(old.setdefault(k, {}), value)
                                elif value is not None:
                                    if old.get(k) is None:
                                        old[k] = value
                                    elif isinstance(value, str) and value != old[k]:
                                        old[k] = str(old[k]) + " " + value

                        merge(records[-1], record)
                    else:
                        records.append(record)
        return records

    def walk(node, path):
        node = _schema(node, schema)
        kind = _type(node)
        if kind == "object":
            return {
                name: walk(child, f"{path}.{name}".strip(".")) for name, child in node.get("properties", {}).items()
            }
        if kind == "array":
            return array(node, path)
        return scalar(node, path)

    return {"data": walk(schema, ""), "evidence": evidence}
