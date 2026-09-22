"""Map complete spacing-delimited rows, repairing only uncertain source cells."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from . import variant_localized
from .variant_multipage import _assign, _leaves
from .variant_tables import _coerce, _index, _question, _schema, _type


@dataclass
class Row:
    page: int
    line: int
    raw: str
    cells: list[str]
    starts: list[int]
    ends: list[int]


def source_rows(document):
    result = []
    pages = document.get("pages") or [{"page_index": 0, "text": document.get("text", "")}]
    for page in pages:
        rows = []
        for line, raw in enumerate(page.get("text", "").splitlines()):
            if not raw.strip() or re.fullmatch(r"[\s|:+\-=]+", raw):
                continue
            if "|" in raw:
                parts = raw.strip().strip("|").split("|")
                cells = [s.strip() for s in parts]
                starts = list(range(len(cells)))
                ends = [s + 1 for s in starts]
            else:
                spans = list(re.finditer(r"\S+(?: \S+)*", raw))
                cells = [s.group().strip() for s in spans]
                starts, ends = [s.start() for s in spans], [s.end() for s in spans]
            rows.append(Row(page.get("page_index", 0), line, raw, cells, starts, ends))
        result.append(rows)
    return result


def _scalar_schema(node, root):
    node = dict(_schema(node, root))
    if _type(node) == "object":
        node["properties"] = {
            k: _scalar_schema(v, root)
            for k, v in node.get("properties", {}).items()
            if _type(_schema(v, root)) != "array"
        }
    return node


def _repair_values(row, field):
    values = list(row.cells)
    if _type(field) in ("number", "integer"):
        values = [m.group() for m in re.finditer(r"\(?[-+]?[$€£¥]?\d[\d,]*(?:\.\d+)?%?\)?", row.raw)] + values
    elif _type(field) == "string":
        values += [a + " " + b for a, b in zip(row.cells, row.cells[1:], strict=False)]
    return list(dict.fromkeys(v for v in values if v))[:240]


def extract(document, schema, client):
    scalar_schema = _scalar_schema(schema, schema)
    initial = variant_localized.extract(document, scalar_schema, client)
    data, evidence = initial["data"], initial.get("evidence", [])
    pages = source_rows(document)

    def arrays(node, path, parent, key):
        node = _schema(node, schema)
        if _type(node) == "object":
            for name, child in node.get("properties", {}).items():
                arrays(child, f"{path}.{name}".strip("."), parent[key], name)
            return
        if _type(node) != "array":
            return
        fields = list(_leaves(node.get("items", {}), schema))
        records = []
        parent[key] = records
        field_descriptions = ", ".join(".".join(names) for names, _ in fields)[:1200]
        if not fields:
            return
        for rows in pages:
            if not rows:
                continue
            # Every row is assessed; page previews never gate unobserved rows.
            selected = []
            for start in range(0, len(rows), 20):
                batch = rows[start : start + 20]
                context = "\n".join(row.raw for row in rows[max(0, start - 3) : start + 23])[:9000]
                answers = client.decide(
                    f"Array: {path}. Fields: {field_descriptions}\nSource:\n{context}",
                    {
                        f"r{i}": {
                            "type": "choice",
                            "instructions": f"Classify this row for {path}: {row.raw}",
                            "criteria": {
                                "c0": "New record",
                                "c1": "Wrapped continuation of preceding record",
                                "c2": "Header, footer, summary, or unrelated text",
                            },
                        }
                        for i, row in enumerate(batch)
                    },
                )
                for i, row in enumerate(batch):
                    kind = _index(answers.get(f"r{i}", {}), 3)
                    if kind in (0, 1):
                        selected.append((row, kind))
            if not selected:
                continue
            counts = Counter(len(row.cells) for row, kind in selected if kind == 0)
            width = counts.most_common(1)[0][0] if counts else max(len(row.cells) for row, _ in selected)
            examples = [row for row, kind in selected if kind == 0 and len(row.cells) == width][:6]
            if not examples:
                examples = [selected[0][0]]
            first_line = examples[0].line
            headers = "\n".join(row.raw for row in rows if first_line - 7 <= row.line <= first_line)
            preview = "\n".join(" | ".join(f"[{c}] {value}" for c, value in enumerate(row.cells)) for row in examples)
            options = [
                f"Column {c}: " + " / ".join(row.cells[c] for row in examples if c < len(row.cells))
                for c in range(width)
            ]
            mapping = client.decide(
                f"Array {path}. Header and source:\n{headers}\nSample complete rows:\n{preview}"[:10000],
                {
                    f"f{f}": _question(
                        f"Which column contains field {path}.{'.'.join(names)}? {field.get('description', '')[:1000]}",
                        options,
                    )
                    for f, (names, field) in enumerate(fields)
                },
            )
            columns = [_index(mapping.get(f"f{f}", {}), width) for f in range(len(fields))]
            prepared, repairs, repair_targets = [], {}, {}
            for row_number, (row, kind) in enumerate(selected):
                values = {}
                prepared.append((row, kind, values))
                for f, (names, field) in enumerate(fields):
                    col = columns[f]
                    raw = row.cells[col] if col is not None and col < len(row.cells) else None
                    value = _coerce(raw, field) if raw else None
                    uncertain = len(row.cells) != width or (bool(raw) and value is None) or kind == 1
                    if uncertain:
                        candidate = _repair_values(row, field)
                        question_id = f"repair{row_number}_{f}"
                        repairs[question_id] = _question(
                            f"Choose {path}.{'.'.join(names)}; absent if not printed. Extract only from this row:\n{row.raw}",  # noqa: E501
                            candidate,
                        )
                        repair_targets[question_id] = (values, f, field, candidate)
                    else:
                        values[f] = (value, raw)
            if repairs:
                # One logical page-level batch; the client splits by byte and question
                # budget. Row-specific source is in each question, not repeated state.
                answers = client.decide(
                    f"Array {path}. Header:\n{headers[:1800]}\nMapped examples:\n{preview[:1800]}", repairs
                )
                for question_id, (values, f, field, candidate) in repair_targets.items():
                    index = _index(answers.get(question_id, {}), len(candidate))
                    raw = candidate[index] if index is not None else None
                    values[f] = (_coerce(raw, field) if raw else None, raw)
            for row, kind, values in prepared:
                record = {} if _type(_schema(node.get("items", {}), schema)) == "object" else None
                for f, (names, _field) in enumerate(fields):
                    value, raw = values[f]
                    record = _assign(record, names, value)
                    if raw:
                        evidence.append(
                            {
                                "path": f"{path}[{max(0, len(records) - (kind == 1))}]." + ".".join(names),
                                "source_text": raw,
                                "context": row.raw,
                                "page_index": row.page,
                                "line_index": row.line,
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

    holder = {"root": data}
    arrays(schema, "", holder, "root")
    return {"data": holder["root"], "evidence": evidence}
