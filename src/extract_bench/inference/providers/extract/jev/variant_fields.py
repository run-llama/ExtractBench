"""Schema-directed extraction by choosing grounded, deterministic text spans.

No generative model or benchmark labels are used. Jev resolves ambiguity among
literal document spans; Python supplies JSON typing and schema reconstruction.
"""

from __future__ import annotations

import json
import re
from typing import Any

from extract_bench.inference.providers.extract.table_codegen.schema_utils import (
    _effective,
    resolve_refs,
)


def _kind(node: dict) -> str:
    value = node.get("type", "object" if "properties" in node else "string")
    return next((v for v in value if v != "null"), "string") if isinstance(value, list) else value


def _tokens(text: str) -> set[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return set(re.findall(r"[a-z0-9]+", text.lower())) - {"the", "of", "a", "in", "and", "to", "is"}


def _lines(document: dict) -> list[dict]:
    pages = document.get("pages") or [{"page_index": 0, "text": document.get("text", "")}]
    result = []
    for page in pages:
        raw = page.get("text", "").splitlines()
        for index, line in enumerate(raw):
            if line.strip():
                result.append(
                    {
                        "text": line.strip(),
                        "raw": line,
                        "page": page.get("page_index", 0),
                        "context": "\n".join(raw[max(0, index - 1) : index + 2]),
                    }
                )
    return result


def _typed(value: str, kind: str) -> Any:
    if kind not in {"integer", "number"}:
        return value.strip()
    clean = re.sub(r"[$€£¥,%\s]", "", value)
    if clean.startswith("(") and clean.endswith(")"):
        clean = "-" + clean[1:-1]
    number = float(clean)
    if kind == "integer":
        if not number.is_integer():
            raise ValueError("Non-integral integer")
        return int(number)
    return number


def candidates(lines: list[dict], path: str, node: dict, limit: int = 48) -> list[dict]:
    """Retrieve label-adjacent spans, cells, dates, numbers and whole lines."""
    kind = _kind(node)
    if "enum" in node:
        return [
            {"value": value, "context": "Schema enumeration", "page": None}
            for value in node["enum"]
            if value is not None
        ]
    if kind == "boolean":
        return [
            {"value": value, "context": "Infer only from explicit document evidence", "page": None}
            for value in (True, False)
        ]
    query = _tokens(path + " " + node.get("description", "") + " " + node.get("title", ""))
    scored = []
    for position, line in enumerate(lines):
        text = line["text"]
        score = len(query & _tokens(line["context"])) / max(1, len(query))
        spans: list[tuple[str, float]] = []
        if kind in {"number", "integer"}:
            spans += [
                (m.group(), 0.2)
                for m in re.finditer(r"(?<![\w.])(?:[$€£¥]\s*)?\(?-?\d[\d,]*(?:\.\d+)?\)?%?(?![\w.])", text)
            ]
        else:
            spans.append((text, 0.0))
            spans += [(cell.strip(), 0.12) for cell in re.split(r"\s{2,}|\t|\|", line["raw"]) if cell.strip()]
            if ":" in text:
                label, value = text.split(":", 1)
                spans.append((value.strip(), 0.35 + len(query & _tokens(label)) / max(1, len(query))))
            spans += [
                (m.group(), 0.1)
                for m in re.finditer(r"\b\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\b|\b[^\s@]+@[^\s@]+\.[^\s@]+\b", text)
            ]
        for span, bonus in spans:
            if not span or len(span) > 700:
                continue
            try:
                value = _typed(span, kind)
            except (ValueError, OverflowError):
                continue
            scored.append(
                (score + bonus, -position, {"value": value, "context": line["context"], "page": line["page"]})
            )
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    result, seen = [], set()
    for _, _, item in scored:
        key = json.dumps(item["value"], ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            result.append(item)
            if len(result) >= limit:
                break
    return result


def _question(prompt: str, options: list[dict]) -> dict:
    return {
        "type": "choice",
        "instructions": prompt,
        "criteria": {
            "none": "Not present / insufficient evidence",
            **{
                f"c{i}": json.dumps(option["value"], ensure_ascii=False) + " | evidence: " + option["context"][:600]
                for i, option in enumerate(options)
            },
        },
    }


def _selected(answer: Any, options: list[dict]) -> dict | None:
    choice = answer.get("choice") if isinstance(answer, dict) else answer
    if isinstance(choice, str) and re.fullmatch(r"c\d+", choice):
        index = int(choice[1:])
        return options[index] if index < len(options) else None
    return None


def extract(document: dict, schema: dict, client) -> dict:
    """Extract nested scalars and row-oriented arrays using batched decisions."""
    schema = resolve_refs(schema)
    lines = _lines(document)
    evidence: list[dict] = []
    leaves: list[tuple[dict | list, str | int, str, dict, list[dict]]] = []
    array_jobs: list[tuple[dict | list, str | int, str, dict, list[dict]]] = []
    root: dict = {}

    def walk(node: dict, parent, key, path: str, local_lines: list[dict], depth: int = 0):
        node = _effective(node)
        kind = _kind(node)
        if depth > 12:
            parent[key] = None
        elif kind == "object":
            parent[key] = {}
            for name, child in node.get("properties", {}).items():
                walk(child, parent[key], name, f"{path}.{name}" if path else name, local_lines, depth + 1)
        elif kind == "array":
            parent[key] = []
            array_jobs.append((parent, key, path, node, local_lines))
        else:
            parent[key] = None
            leaves.append((parent, key, path, node, local_lines))

    walk(schema, root, "data", "", lines)
    # Arrays are proposed from physical rows. A separate relevance decision
    # suppresses headers, totals, narrative text and unrelated rows.
    for parent, key, path, node, local_lines in array_jobs[:32]:
        item = _effective(node.get("items", {}))
        row_lines = [line for line in local_lines if len(re.split(r"\s{2,}|\t|\|", line["raw"].strip())) > 1]
        if not row_lines:
            row_lines = local_lines
        row_lines = row_lines[:48]
        questions = {
            f"row{i}": {
                "type": "choice",
                "instructions": f"Does this row contain one actual item of {path}? "
                f"Array description: {node.get('description', '')}. Item schema: {json.dumps(item)}. "
                f"Row: {line['text']}. Context: {line['context']}. Exclude headers and totals.",
                "criteria": {"yes": "Yes, one data item", "no": "No"},
            }
            for i, line in enumerate(row_lines)
        }
        answers = client.decide({"document": document.get("text", "")[:18000]}, questions) if questions else {}
        for i, line in enumerate(row_lines):
            answer = answers.get(f"row{i}", {})
            if (answer.get("choice") if isinstance(answer, dict) else answer) == "yes":
                index = len(parent[key])
                parent[key].append(None)
                walk(item, parent[key], index, f"{path}[{index}]", [line])

    questions, jobs = {}, []
    for parent, key, path, node, local_lines in leaves:
        options = candidates(local_lines, path, node)
        identifier = f"field{len(jobs)}"
        questions[identifier] = _question(
            f"Select the exact value for {path}. "
            f"Description: {node.get('description', '')}. Type: {_kind(node)}. "
            "Use the candidate evidence to distinguish similar fields. Select none if missing.",
            options,
        )
        jobs.append((identifier, parent, key, path, options))
    answers = client.decide({"document": document.get("text", "")[:18000]}, questions) if questions else {}
    for identifier, parent, key, path, options in jobs:
        selected = _selected(answers.get(identifier), options)
        if selected is not None:
            parent[key] = selected["value"]
            evidence.append({"path": path, "page_index": selected["page"], "text": selected["context"]})
    return {"data": root["data"], "evidence": evidence}
