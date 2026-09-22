"""Layout-first extraction: Jev selects tables/columns, Python copies cells.

All returned strings originate in parsed source text or explicit schema enums.
No document labels or benchmark ground truth are consulted.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass
class Table:
    rows: list[list[str]]
    lines: list[int]

    @property
    def width(self):
        return max(map(len, self.rows), default=0)

    def preview(self):
        return "\n".join(f"row {i}: " + " | ".join(row) for i, row in enumerate(self.rows[:12]))


def discover_tables(text: str) -> list[Table]:
    """Keep contiguous aligned regions; preserve empty pipe-delimited cells."""
    tables, rows, indices = [], [], []

    def flush():
        nonlocal rows, indices
        if len(rows) >= 2:
            tables.append(Table(rows, indices))
        rows, indices = [], []

    for i, raw in enumerate(text.splitlines()):
        line = raw.strip()
        if re.fullmatch(r"[\s|:+\-=]+", line or "x"):
            continue
        if "|" in line:
            cells = [part.strip() for part in line.strip("|").split("|")]
        else:
            cells = re.split(r"\t+| {2,}", line)
        if len(cells) < 2:
            flush()
            continue
        # A width change often marks a different table, but tolerate sparse rows.
        if rows and abs(len(cells) - Counter(map(len, rows)).most_common(1)[0][0]) > 2:
            flush()
        rows.append(cells)
        indices.append(i)
    flush()
    return tables


def _schema(node: dict, root: dict) -> dict:
    seen = set()
    while "$ref" in node and node["$ref"] not in seen:
        ref = node["$ref"]
        seen.add(ref)
        if not ref.startswith("#/"):
            break
        target = root
        for key in ref[2:].split("/"):
            target = target.get(key.replace("~1", "/").replace("~0", "~"), {})
        node = {**target, **{k: v for k, v in node.items() if k != "$ref"}}
    for key in ("anyOf", "oneOf"):
        if key in node:
            options = [x for x in node[key] if x.get("type") != "null"]
            if options:
                node = {**node, **options[0]}
    return node


def _type(node: dict):
    kind = node.get("type", "object" if "properties" in node else "string")
    return next((x for x in kind if x != "null"), "string") if isinstance(kind, list) else kind


def _coerce(value: str | None, node: dict):
    if value is None:
        return None
    value = value.strip()
    if _type(node) in ("number", "integer"):
        cleaned = re.sub(r"[$€£¥,%\s]", "", value).replace(",", "")
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = "-" + cleaned[1:-1]
        try:
            result = float(cleaned)
            return int(result) if _type(node) == "integer" and result.is_integer() else result
        except ValueError:
            return None
    if _type(node) == "boolean":
        if value.lower() in ("yes", "true", "1", "checked"):
            return True
        if value.lower() in ("no", "false", "0", "unchecked"):
            return False
        return None
    return value


def _question(instructions: str, choices: list[str]) -> dict:
    return {
        "type": "choice",
        "instructions": instructions,
        "criteria": {**{f"c{i}": c for i, c in enumerate(choices)}, "none": "Not present / no supported match"},
    }


def _index(answer: dict, length: int) -> int | None:
    choice = answer.get("choice", "")
    if isinstance(choice, str) and re.fullmatch(r"c\d+", choice):
        index = int(choice[1:])
        return index if index < length else None
    return None


def _candidates(text: str, field: str, node: dict) -> list[str]:
    terms = set(
        re.findall(
            r"[a-z]{3,}", re.sub(r"([a-z])([A-Z])", r"\1 \2", field).lower() + " " + node.get("description", "").lower()
        )
    )
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    ranked = sorted(enumerate(lines), key=lambda item: -sum(term in item[1].lower() for term in terms))[:50]
    values = []
    for index, line in ranked:
        parts = re.split(r"\s{2,}|\t|\s*:\s*|\s*\|\s*", line)
        values.extend(parts + [line])
        if index + 1 < len(lines):
            values.append(lines[index + 1])
        if _type(node) in ("number", "integer"):
            values.extend(re.findall(r"\(?[-+]?[$€£]?\d[\d,]*(?:\.\d+)?\)?%?", line))
    if "enum" in node:
        values = [str(v) for v in node["enum"] if v is not None] + values
    return list(dict.fromkeys(v.strip() for v in values if v.strip()))[:200]


def extract(document: dict, schema: dict, client) -> dict:
    text = document.get("text") or "\n".join(p.get("text", "") for p in document.get("pages", []))
    tables = discover_tables(text)
    evidence: list[dict[str, Any]] = []
    scalar_jobs = []
    table_jobs = []

    def plan(node, path, container, key):
        node = _schema(node, schema)
        kind = _type(node)
        if kind == "object":
            container[key] = {}
            for name, child in node.get("properties", {}).items():
                plan(child, f"{path}.{name}".strip("."), container[key], name)
        elif kind == "array":
            container[key] = []
            table_jobs.append((path, node, container, key))
        else:
            scalar_jobs.append((path, node, container, key))

    holder = {}
    plan(schema, "", holder, "root")
    questions, candidates = {}, {}
    for i, (path, node, _, _) in enumerate(scalar_jobs):
        values = _candidates(text, path, node)
        candidates[f"s{i}"] = values
        questions[f"s{i}"] = _question(
            f"Choose the exact source value for field {path}. Schema: {node}. Do not choose the field label itself.",
            values,
        )
    for i, (path, node, _, _) in enumerate(table_jobs):
        questions[f"t{i}"] = _question(
            f"Choose the table containing records for array {path}. Schema: {node}", [t.preview() for t in tables[:100]]
        )
    answers = client.decide(text[:60000], questions) if questions else {}
    for i, (path, node, container, key) in enumerate(scalar_jobs):
        values = candidates[f"s{i}"]
        index = _index(answers.get(f"s{i}", {}), len(values))
        container[key] = _coerce(values[index], node) if index is not None else None
        if index is not None:
            evidence.append({"path": path, "source_text": values[index]})

    for i, (path, node, container, key) in enumerate(table_jobs):
        index = _index(answers.get(f"t{i}", {}), min(len(tables), 100))
        if index is None:
            continue
        table = tables[index]
        item = _schema(node.get("items", {}), schema)
        fields = []

        def leaves(current, prefix, fields=fields):
            current = _schema(current, schema)
            if _type(current) == "object":
                for name, child in current.get("properties", {}).items():
                    leaves(child, prefix + [name])
            elif _type(current) != "array":
                fields.append((prefix, current))

        leaves(item, [])
        columns = [
            "Column " + str(c) + ": " + " / ".join(row[c] for row in table.rows[:10] if c < len(row))
            for c in range(table.width)
        ]
        mapping_questions = {
            f"c{j}": _question(f"Which column supplies {path}.{'.'.join(names)}? Field schema: {field}", columns)
            for j, (names, field) in enumerate(fields)
        }
        mapping_questions.update(
            {
                f"r{r}": _question(
                    f"Classify row {r} for array {path}: " + " | ".join(row),
                    ["A data record belonging to this array", "A header, subtotal, total, footnote, or unrelated row"],
                )
                for r, row in enumerate(table.rows)
            }
        )
        decisions = client.decide(f"Extract array {path}. Schema: {node}\nTable:\n{table.preview()}", mapping_questions)
        mapping = [_index(decisions.get(f"c{j}", {}), len(columns)) for j in range(len(fields))]
        for r, row in enumerate(table.rows):
            if _index(decisions.get(f"r{r}", {}), 2) != 0:
                continue
            record = {} if _type(item) == "object" else None
            for j, (names, field) in enumerate(fields):
                column = mapping[j]
                raw = row[column] if column is not None and column < len(row) else None
                value = _coerce(raw, field)
                if not names:
                    record = value
                else:
                    dest = record
                    for name in names[:-1]:
                        dest = dest.setdefault(name, {})
                    dest[names[-1]] = value
                if raw is not None:
                    evidence.append(
                        {
                            "path": f"{path}[{len(container[key])}]." + ".".join(names),
                            "source_text": raw,
                            "line_index": table.lines[r],
                        }
                    )
            container[key].append(record)
    return {"data": holder["root"], "evidence": evidence}
