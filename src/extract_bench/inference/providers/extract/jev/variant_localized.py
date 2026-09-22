"""Localized source-span decisions with document-wide lexical region retrieval."""

from __future__ import annotations

import json
import math
import re
from collections import Counter

from .variant_hierarchical import _resolve, _tokens


def _regions(document: dict) -> list[dict]:
    regions = []
    pages = document.get("pages") or [{"page_index": 0, "text": document.get("text", "")}]
    for page in pages:
        lines = []
        for raw in page.get("text", "").splitlines():
            if not raw.strip():
                continue
            # Split pathological OCR lines with overlap rather than dropping their tail.
            raw = raw.rstrip()
            lines.extend(raw[start : start + 300] for start in range(0, len(raw), 270))
        # Overlap captures labels whose value is printed on the following line.
        for start in range(0, len(lines), 5):
            content = "\n".join(lines[max(0, start - 1) : start + 7])
            regions.append(
                {
                    "text": content,
                    "page_index": page.get("page_index", 0),
                    "line_index": start,
                    "tokens": _tokens(content),
                }
            )
    return regions


def _retrieve(regions: list[dict], path: str, node: dict, limit: int = 8) -> list[dict]:
    frequency = Counter(token for region in regions for token in region["tokens"])
    main = _tokens(path + " " + node.get("title", ""))
    extra = _tokens(node.get("description", "")) - main
    count = max(len(regions), 1)

    def score(region):
        return sum(
            (3 if token in main else 1) * math.log(1 + count / (1 + frequency[token]))
            for token in region["tokens"] & (main | extra)
        )

    ranked = sorted(enumerate(regions), key=lambda pair: (-score(pair[1]), pair[0]))
    # Do not sort the shortlist back into source order: that silently favors early pages.
    chosen = [region for _, region in ranked[:limit]]
    return chosen


def _values(text: str, node: dict) -> list[str]:
    values = []
    numeric = node.get("type") in ("number", "integer")
    lines = text.splitlines()
    for line in lines:
        if numeric:
            pieces = re.findall(r"\(?[-+]?[$€£]?\d[\d,]*(?:\.\d+)?%?\)?", line)
            # OCR sometimes separates a final digit after a thousands comma.
            pieces += re.findall(r"[-+]?\d{1,3},\d{2} \d\b", line)
        else:
            pieces = re.split(r"\t+| {3,}|\s*\|\s*", line.strip())
            # Cell suffixes handle a label and value printed without a wide gap.
            pieces += [x.split(":", 1)[1].strip() for x in pieces if ":" in x]
            pieces += re.findall(r"\b\d{1,4}[/-]\d{1,2}[/-]\d{1,4}\b|\b\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?\b", line)
            pieces += [line.strip()]
        for value in pieces:
            if value and value not in values:
                values.append(value)
    if not numeric:
        # Add literal word-boundary substrings; keep original internal spacing.
        for line in lines:
            words = list(re.finditer(r"\S+", line))
            for length in (1, 2, 3, 4, 5, 6):
                for i in range(len(words) - length + 1):
                    value = line[words[i].start() : words[i + length - 1].end()]
                    if value not in values:
                        values.append(value)
    return values[:240]


def _normalize(value, node: dict):
    if not isinstance(value, str):
        return value
    if node.get("type") in ("number", "integer"):
        cleaned = re.sub(r"[$€£¥,%\s]", "", value)
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = "-" + cleaned[1:-1]
        try:
            number = float(cleaned)
            if not math.isfinite(number):
                return None
            return (
                int(number)
                if node.get("type") == "integer" and number.is_integer()
                else number
                if node.get("type") == "number"
                else None
            )
        except ValueError:
            return None
    # Layout spaces separate words, but whitespace does not change semantic text.
    return re.sub(r"\s+", " ", value).strip()


def _question(instructions: str, values: list) -> dict:
    return {
        "type": "choice",
        "instructions": instructions,
        "criteria": {
            **{f"v{i}": json.dumps(value, ensure_ascii=False) for i, value in enumerate(values)},
            "none": "No value supported by this source; null",
        },
    }


def _picked(answer: dict, values: list):
    choice = answer.get("choice", "")
    if isinstance(choice, str) and re.fullmatch(r"v\d+", choice):
        index = int(choice[1:])
        if index < len(values):
            return values[index]
    return None


def extract(document: dict, schema: dict, client) -> dict:
    regions = _regions(document)
    jobs, arrays, evidence = [], [], []
    holder = {}

    def plan(node, path, dest, key, depth=0):
        node = _resolve(node, schema)
        kind = node.get("type", "object" if "properties" in node else "string")
        if depth > 20:
            dest[key] = None
        elif kind == "object":
            dest[key] = {}
            for name, child in node.get("properties", {}).items():
                plan(child, f"{path}.{name}".strip("."), dest[key], name, depth + 1)
        elif kind == "array":
            dest[key] = []
            arrays.append((path, node, dest, key))
        else:
            dest[key] = None
            jobs.append((path, node, dest, key))

    plan(schema, "", holder, "root")
    # Region selection is batched, while each question carries its own local evidence.
    for start in range(0, len(jobs), 8):
        batch = jobs[start : start + 8]
        questions, candidates = {}, {}
        for i, (path, node, _, _) in enumerate(batch):
            candidates[i] = _retrieve(regions, path, node)
            choices = [
                f"Page {r['page_index'] + 1}, lines around {r['line_index']}:\n{r['text']}" for r in candidates[i]
            ]
            questions[str(i)] = _question(
                f"Find the source region containing the VALUE (not merely a mention) of {path}. Field schema: {json.dumps(node)}. A blank checkbox is still relevant evidence.",  # noqa: E501
                choices,
            )
        answers = (
            client.decide(
                "Select source regions for document extraction. Do not infer values from examples in field descriptions.",  # noqa: E501
                questions,
            )
            if questions
            else {}
        )
        questions, choices, sources = {}, {}, {}
        for i, (path, node, _dest, _key) in enumerate(batch):
            descriptions = [
                f"Page {r['page_index'] + 1}, lines around {r['line_index']}:\n{r['text']}" for r in candidates[i]
            ]
            selected = _picked(answers.get(str(i), {}), descriptions)
            # Booleans must see evidence even if the region stage abstained.
            if selected is None and node.get("type") == "boolean":
                selected = "\n".join(descriptions[:2])
            if selected is None:
                continue
            sources[i] = selected
            if node.get("type") == "boolean":
                values = [True, False]
            elif "enum" in node:
                values = [v for v in node["enum"] if v is not None]
            else:
                # Omit metadata so page/line numbers cannot become field candidates.
                raw = selected.split("\n", 1)[1] if "\n" in selected else selected
                values = _values(raw, node)
            choices[i] = values
            questions[str(i)] = _question(
                f"Extract {path}. Field schema: {json.dumps(node)}\nSOURCE:\n{selected}\nChoose only the actual field value, excluding labels, adjacent columns, line numbers, and explanatory examples. Honor any explicit blank/false/null rules. Do not treat an unmarked checkbox as checked.",  # noqa: E501
                values,
            )
        answers = (
            client.decide("Extract exact source spans or schema-defined categories from local evidence.", questions)
            if questions
            else {}
        )
        for i, (path, node, dest, key) in enumerate(batch):
            value = _picked(answers.get(str(i), {}), choices.get(i, []))
            if value is None and node.get("type") == "boolean":
                description = node.get("description", "").lower()
                if "never return null" in description or "false otherwise" in description:
                    value = False
            dest[key] = _normalize(value, node)
            if value is not None:
                evidence.append({"path": path, "source_text": value, "context": sources.get(i, "")})
    if arrays:
        from . import variant_tables

        # Pass full documents; the layout variant classifies all rows of selected tables.
        array_schema = {
            "type": "object",
            "$defs": schema.get("$defs", {}),
            "definitions": schema.get("definitions", {}),
            "properties": {
                f"a{i}": {**node, "description": f"Array field {path}. " + node.get("description", "")}
                for i, (path, node, _, _) in enumerate(arrays)
            },
        }
        array_result = variant_tables.extract(document, array_schema, client)
        for i, (_path, _, dest, key) in enumerate(arrays):
            dest[key] = array_result["data"].get(f"a{i}", [])
        for item in array_result.get("evidence", []):
            item = dict(item)
            for i, (path, _, _, _) in enumerate(arrays):
                if item.get("path", "").startswith(f"a{i}["):
                    item["path"] = path + item["path"][len(f"a{i}") :]
                    break
            evidence.append(item)
    return {"data": holder["root"], "evidence": evidence}
