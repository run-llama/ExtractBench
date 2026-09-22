"""Grounded extraction by successive source-line and source-span decisions.

Jev only selects choices: every free-text value originates in the parsed input.
No reference answers or document-specific rules are used.
"""

from __future__ import annotations

import json
import re
from typing import Any

_STOP = set("the a an of in for and to is are as from on with value field".split())


def _resolve(node: dict, root: dict, depth: int = 0) -> dict:
    if depth > 20:
        return {"type": "string"}
    if "$ref" in node:
        target = root
        for part in node["$ref"].removeprefix("#/").split("/"):
            target = target.get(part.replace("~1", "/").replace("~0", "~"), {})
        return _resolve({**target, **{k: v for k, v in node.items() if k != "$ref"}}, root, depth + 1)
    branches = node.get("anyOf", node.get("oneOf", []))
    if branches:
        chosen = next((b for b in branches if b.get("type") != "null"), {})
        return _resolve({**{k: v for k, v in node.items() if k not in ("anyOf", "oneOf")}, **chosen}, root, depth + 1)
    result = dict(node)
    if isinstance(result.get("type"), list):
        result["type"] = next((t for t in result["type"] if t != "null"), "string")
    return result


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", text).lower())) - _STOP


def _lines(document: dict) -> list[dict]:
    result = []
    pages = document.get("pages") or [{"text": document.get("text", ""), "page_index": 0}]
    for page in pages:
        for line in page.get("text", "").splitlines():
            if line.strip():
                result.append(
                    {"text": line.strip(), "page_index": page.get("page_index", 0), "line_index": len(result)}
                )
    return result


def _rank(lines: list[dict], description: str, limit: int = 48) -> list[dict]:
    query = _tokens(description)
    ranked = sorted(
        enumerate(lines), key=lambda pair: (-sum(1 for t in query if t in _tokens(pair[1]["text"])), pair[0])
    )
    indices = {i for i, _ in ranked[: max(1, limit // 2)]}
    for i in list(indices):
        indices.update(j for j in (i - 1, i + 1) if 0 <= j < len(lines))
    return [lines[i] for i in sorted(indices)][:limit]


def _normalize(value: str, node: dict) -> Any:
    typ = node.get("type", "string")
    value = value.strip()
    if typ in ("number", "integer"):
        raw = re.sub(r"[^\d.\-+eE]", "", value.replace(",", ""))
        try:
            number = float(raw)
            if value.startswith("(") and value.endswith(")"):
                number = -number
            if typ == "integer":
                return int(number) if number.is_integer() else None
            return number
        except (ValueError, OverflowError):
            return None
    return value


def _spans(texts: list[str], node: dict) -> list[Any]:
    if "enum" in node:
        return list(node["enum"])
    if node.get("type") == "boolean":
        return [True, False]
    candidates = []
    for text in texts:
        if node.get("type") in ("integer", "number"):
            spans = re.findall(r"\(?[-+]?[$€£]?\d[\d,]*(?:\.\d+)?%?\)?", text)
        else:
            spans = [text]
            spans += re.split(r"\s{2,}|\t|\s\|\s", text)
            if ":" in text:
                spans += [text.split(":", 1)[1].strip()]
            words = text.split()
            # Include long cell values before the contiguous short spans.
            spans += [" ".join(words[i:]) for i in range(1, min(len(words), 8))]
            spans += [
                " ".join(words[i : i + n]) for n in range(1, min(7, len(words)) + 1) for i in range(len(words) - n + 1)
            ]
        for span in spans:
            value = _normalize(span, node)
            if value not in (None, "") and value not in candidates:
                candidates.append(value)
    return candidates[:160]


def _ask(client, state: str, questions: dict) -> dict:
    if not questions:
        return {}
    source_labels = dict(re.findall(r"^(L\d+): (.*)$", state, re.MULTILINE))
    formatted = {
        key: {
            "type": "choice",
            "instructions": question["question"],
            "criteria": {
                choice: question.get("labels", {}).get(
                    choice,
                    source_labels.get(
                        choice,
                        {
                            "NONE": "No supported value is present",
                            "YES": "This line contains a distinct requested record",
                            "NO": "This line is not a requested record",
                        }.get(choice, choice),
                    ),
                )
                for choice in question["choices"]
            },
        }
        for key, question in questions.items()
    }
    return client.decide(state, formatted)


def _choice(answers: dict, key: str) -> str | None:
    answer = answers.get(key, {})
    return str(answer.get("choice")) if answer.get("choice") is not None else None


def extract(document: dict, schema: dict, client) -> dict:
    """Extract a JSON object through two batched decisions per scalar layer."""
    lines = _lines(document)
    evidence = []
    counter = 0

    def scalar_batch(fields: list[tuple[str, dict]], source: list[dict]) -> dict:
        nonlocal counter
        questions, candidates, descriptions = {}, {}, {}
        for path, node in fields:
            desc = f"{path}: {node.get('description', '')} (type {node.get('type', 'string')})"
            selected = _rank(source, desc)
            key = f"line_{counter}"
            counter += 1
            candidates[key] = (path, node, selected)
            descriptions[key] = desc
            questions[key] = {
                "question": f"Which source line contains the value of {desc}? Select NONE if absent.",
                "choices": ["NONE"] + [f"L{line['line_index']}" for line in selected],
            }
        state = "Source document lines (use only supported information):\n" + "\n".join(
            f"L{x['line_index']}: {x['text']}" for x in source
        )
        answers = _ask(client, state, questions)
        result = {path: None for path, _ in fields}
        span_questions, span_data = {}, {}
        for key, (path, node, selected) in candidates.items():
            chosen = _choice(answers, key)
            line = next((line for line in selected if f"L{line['line_index']}" == chosen), None)
            if line is None:
                continue
            nearby = [line["text"]] + [x["text"] for x in source if abs(x["line_index"] - line["line_index"]) == 1]
            spans = _spans(nearby, node)
            span_key = key.replace("line_", "span_")
            span_data[span_key] = (path, line, spans)
            span_questions[span_key] = {
                "question": f"Choose the exact value for {descriptions[key]}. Source: {json.dumps(nearby)}. Options: "
                + json.dumps({f"V{i}": v for i, v in enumerate(spans)}, ensure_ascii=False)
                + ". NONE means unsupported.",
                "choices": ["NONE"] + [f"V{i}" for i in range(len(spans))],
                "labels": {f"V{i}": json.dumps(value, ensure_ascii=False) for i, value in enumerate(spans)},
            }
        answers = _ask(client, state, span_questions)
        for key, (path, line, spans) in span_data.items():
            chosen = _choice(answers, key)
            if chosen and re.fullmatch(r"V\d+", chosen) and int(chosen[1:]) < len(spans):
                result[path] = spans[int(chosen[1:])]
                evidence.append(
                    {
                        "field": path,
                        "page_index": line["page_index"],
                        "line_index": line["line_index"],
                        "text": line["text"],
                    }
                )
        return result

    def walk(node: dict, path: str, source: list[dict], depth: int = 0):
        nonlocal counter
        node = _resolve(node, schema)
        typ = node.get("type", "object" if "properties" in node else "string")
        if depth > 12:
            return None
        if typ == "object":
            result, scalar_fields, scalar_names = {}, [], {}
            for name, child in node.get("properties", {}).items():
                child = _resolve(child, schema)
                child_path = f"{path}.{name}" if path else name
                if child.get("type", "object" if "properties" in child else "string") in ("object", "array"):
                    result[name] = walk(child, child_path, source, depth + 1)
                else:
                    scalar_fields.append((child_path, child))
                    scalar_names[child_path] = name
            for field, value in scalar_batch(scalar_fields, source).items():
                result[scalar_names[field]] = value
            return result
        if typ == "array":
            item = _resolve(node.get("items", {"type": "string"}), schema)
            selected = _rank(source, f"{path} {node.get('description', '')} {json.dumps(item)}", 64)
            questions, keys = {}, {}
            for line in selected:
                key = f"row_{counter}"
                counter += 1
                keys[key] = line
                questions[key] = {
                    "question": f"Does line L{line['line_index']} contain a distinct data record for array {path} ({node.get('description', '')}) with item schema {json.dumps(item)}? Exclude headers and totals unless requested.",  # noqa: E501
                    "choices": ["YES", "NO"],
                }
            state = "\n".join(f"L{x['line_index']}: {x['text']}" for x in source)
            answers = _ask(client, state, questions)
            rows = [line for key, line in keys.items() if _choice(answers, key) == "YES"]
            return [walk(item, f"{path}[{i}]", [line], depth + 1) for i, line in enumerate(rows)]
        return scalar_batch([(path, node)], source).get(path)

    return {"data": walk(schema, "", lines), "evidence": evidence}
