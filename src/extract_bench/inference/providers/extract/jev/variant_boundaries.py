"""Jev selects a source region, then literal start and end word boundaries."""

from __future__ import annotations

import json
import re

from .variant_hierarchical import _resolve
from .variant_localized import _normalize, _picked, _question, _regions, _retrieve


def _bounded_regions(document):
    result = []
    for region in _regions(document):
        words = list(re.finditer(r"\S+", region["text"]))
        if len(words) <= 220:
            result.append(region)
        else:
            from .variant_hierarchical import _tokens

            for start in range(0, len(words), 190):
                text = region["text"][words[start].start() : words[min(start + 219, len(words) - 1)].end()]
                result.append({**region, "text": text, "tokens": _tokens(text)})
    return result


def _index(answer, length):
    match = re.fullmatch(r"w(\d+)", answer.get("choice", ""))
    return int(match[1]) if match and int(match[1]) < length else None


def _array_schema(node, root):
    node = _resolve(node, root)
    kind = node.get("type", "object" if "properties" in node else "string")
    if kind == "array":
        return node
    if kind != "object":
        return None
    properties = {}
    for name, child in node.get("properties", {}).items():
        retained = _array_schema(child, root)
        if retained is not None:
            properties[name] = retained
    if not properties:
        return None
    return {**node, "properties": properties, "required": [k for k in node.get("required", []) if k in properties]}


def _merge(dest, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(dest.get(key), dict):
            _merge(dest[key], value)
        else:
            dest[key] = value


def extract(document: dict, schema: dict, client) -> dict:
    regions = _bounded_regions(document)
    jobs, evidence, holder = [], [], {}

    def walk(node, path, dest, key, depth=0):
        node = _resolve(node, schema)
        kind = node.get("type", "object" if "properties" in node else "string")
        if depth > 20:
            dest[key] = None
        elif kind == "object":
            dest[key] = {}
            for name, child in node.get("properties", {}).items():
                walk(child, f"{path}.{name}".strip("."), dest[key], name, depth + 1)
        elif kind == "array":
            dest[key] = []
        else:
            dest[key] = None
            jobs.append((path, node, dest, key))

    walk(schema, "", holder, "data")
    for offset in range(0, len(jobs), 8):
        batch = jobs[offset : offset + 8]
        retrieved, questions = {}, {}
        for i, (path, node, _, _) in enumerate(batch):
            retrieved[i] = _retrieve(regions, path, node, 8)
            descriptions = [f"Page {r['page_index'] + 1}:\n{r['text']}" for r in retrieved[i]]
            questions[str(i)] = _question(
                f"Which source region contains the actual value for {path}? Field schema: {json.dumps(node)}. Match the correct entity and field; checkbox labels and blank fields still provide evidence.",  # noqa: E501
                descriptions,
            )
        answers = (
            client.decide(
                "Locate each requested field using source text. Schema examples are not source values.", questions
            )
            if questions
            else {}
        )
        selected, words_by_id, questions, categories = {}, {}, {}, {}
        for i, (path, node, _, _) in enumerate(batch):
            descriptions = [f"Page {r['page_index'] + 1}:\n{r['text']}" for r in retrieved[i]]
            chosen = _picked(answers.get(str(i), {}), descriptions)
            if chosen is None and node.get("type") == "boolean" and descriptions:
                chosen = descriptions[0]
            if chosen is None:
                continue
            region = retrieved[i][descriptions.index(chosen)]
            selected[i] = region
            prompt = f"Field {path}. Schema: {json.dumps(node)}\nSOURCE:\n{region['text']}\n"
            if node.get("type") == "boolean" or "enum" in node:
                values = [v for v in node.get("enum", [True, False]) if v is not None]
                categories[i] = values
                questions[str(i)] = _question(
                    prompt
                    + "Choose the source-supported value. Respect explicit rules for blank, unchecked, false, and null. A label alone does not prove its checkbox is marked.",  # noqa: E501
                    values,
                )
                continue
            words = list(re.finditer(r"\S+", region["text"]))
            words_by_id[i] = words
            criteria = {"none": "The field value is absent, blank, or cannot be read"}
            for j, word in enumerate(words):
                after = region["text"][word.end() : words[min(j + 5, len(words) - 1)].end()]
                before = region["text"][words[max(0, j - 2)].start() : word.start()]
                criteria[f"w{j}"] = f"Word {j}: {before} [START {word.group()}] {after}"
            questions[str(i)] = {
                "type": "choice",
                "instructions": prompt
                + "Choose the FIRST word of the exact value. Exclude field labels, neighboring columns, and unrelated text. The source span can have any length within this region; an end boundary will be selected next.",  # noqa: E501
                "criteria": criteria,
            }
        answers = (
            client.decide("Choose categorical values or literal start boundaries, never generate text.", questions)
            if questions
            else {}
        )
        starts, questions = {}, {}
        for i, (path, node, dest, key) in enumerate(batch):
            if i in categories:
                value = _picked(answers.get(str(i), {}), categories[i])
                description = node.get("description", "").lower()
                if (
                    value is None
                    and node.get("type") == "boolean"
                    and ("never return null" in description or "false otherwise" in description)
                ):
                    value = False
                dest[key] = value
                if value is not None:
                    evidence.append(
                        {
                            "path": path,
                            "page_index": selected[i]["page_index"],
                            "text": selected[i]["text"],
                            "value": value,
                        }
                    )
                continue
            if i not in words_by_id:
                continue
            words = words_by_id[i]
            start = _index(answers.get(str(i), {}), len(words))
            if start is None:
                continue
            starts[i] = start
            region = selected[i]
            criteria = {"none": "The selected beginning is not a supported field value"}
            for j in range(start, len(words)):
                context = region["text"][words[max(start, j - 4)].start() : words[j].end()]
                after = region["text"][words[j].end() : words[min(j + 2, len(words) - 1)].end()]
                criteria[f"w{j}"] = f"End after word {j}: {context} [END] {after}"
            questions[str(i)] = {
                "type": "choice",
                "instructions": f"Field {path}. Schema: {json.dumps(node)}\nSOURCE:\n{region['text']}\nThe value begins at word {start}, {words[start].group()!r}. Choose its LAST word. Include the complete name/address/value but exclude adjacent columns and labels. For a one-word value the end is the same word as the start.",  # noqa: E501
                "criteria": criteria,
            }
        answers = client.decide("Select inclusive source-span end boundaries.", questions) if questions else {}
        for i, (path, node, dest, key) in enumerate(batch):
            if i not in starts:
                continue
            words = words_by_id[i]
            end = _index(answers.get(str(i), {}), len(words))
            start = starts[i]
            if end is None or end < start:
                continue
            region = selected[i]
            raw = region["text"][words[start].start() : words[end].end()]
            dest[key] = _normalize(raw, node)
            if dest[key] is not None:
                evidence.append(
                    {
                        "path": path,
                        "page_index": region["page_index"],
                        "text": raw,
                        "context": region["text"],
                        "start_char": words[start].start(),
                        "end_char": words[end].end(),
                    }
                )
    arrays = _array_schema(schema, schema)
    if arrays is not None:
        from .variant_multipage import extract as extract_arrays

        arrays["$defs"] = schema.get("$defs", {})
        arrays["definitions"] = schema.get("definitions", {})
        result = extract_arrays(document, arrays, client)
        if isinstance(holder["data"], dict):
            _merge(holder["data"], result["data"])
        else:
            holder["data"] = result["data"]
        evidence.extend(result.get("evidence", []))
    return {"data": holder["data"], "evidence": evidence}
