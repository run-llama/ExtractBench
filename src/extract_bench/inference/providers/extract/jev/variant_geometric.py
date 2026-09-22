"""Geometry-reconstructed spans with local Jev decisions and column arrays."""

from __future__ import annotations

import json
from statistics import median

from extract_bench.inference.providers.extract.jev import variant_fields as fields
from extract_bench.inference.providers.extract.jev import variant_tables as tables
from extract_bench.inference.providers.extract.table_codegen.schema_utils import _effective, resolve_refs


def geometric_lines(document: dict) -> list[dict]:
    """Reassemble physical baselines before reading text; retain source geometry."""
    result = []
    for page in document.get("pages") or [{"text": document.get("text", "")}]:
        words = [w for w in page.get("words", []) if str(w.get("text", "")).strip()]
        page_index = page.get("page_index", 0)
        if not words:
            result.extend(fields._lines({"pages": [page]}))
            continue
        height = median(max(float(w.get("height", 1)), 1) for w in words)
        rows: list[dict] = []
        for word in sorted(words, key=lambda w: (w["y"] + w.get("height", height) / 2, w["x"])):
            center = word["y"] + word.get("height", height) / 2
            row = next((r for r in reversed(rows[-4:]) if abs(r["center"] - center) <= height * 0.55), None)
            if row is None:
                row = {"center": center, "words": []}
                rows.append(row)
            row["words"].append(word)
            row["center"] = median(w["y"] + w.get("height", height) / 2 for w in row["words"])
        for row in rows:
            ordered = sorted(row["words"], key=lambda w: w["x"])
            cells, current = [], []
            for word in ordered:
                if current and word["x"] - (current[-1]["x"] + current[-1].get("width", 0)) > height * 1.5:
                    cells.append(" ".join(w["text"].strip() for w in current))
                    current = []
                current.append(word)
            if current:
                cells.append(" ".join(w["text"].strip() for w in current))
            result.append(
                {
                    "text": " ".join(cells),
                    "raw": "   ".join(cells),
                    "page": page_index,
                    "words": ordered,
                    "y": row["center"],
                    "cells": cells,
                }
            )
    for i, row in enumerate(result):
        row["context"] = "\n".join(r["raw"] for r in result[max(0, i - 1) : i + 2] if r["page"] == row["page"])
    return result


def span_candidates(lines: list[dict], path: str, node: dict, limit: int = 96) -> list[dict]:
    kind = fields._kind(node)
    if "enum" in node or kind == "boolean":
        options = fields.candidates(lines, path, node)
        context = "\n".join(line["raw"] for line in _relevant(lines, path, node)[:14])
        return [{**option, "context": context} for option in options]
    query = fields._tokens(path.split(".")[-1]) | fields._tokens(node.get("title", ""))
    description = fields._tokens(node.get("description", ""))
    expanded = []
    for row in lines:
        label_score = len(query & fields._tokens(row["text"])) / max(1, len(query))
        near_score = len(query & fields._tokens(row["context"])) / max(1, len(query))
        desc_score = len(description & fields._tokens(row["context"])) / max(1, len(description))
        base = fields.candidates([row], path, node, limit=80)
        for option in base:
            option = {**option, "context": f"SOURCE ROW: {row['raw']}\nNearby: {row['context']}"}
            expanded.append((label_score * 3 + near_score + desc_score, option))
        if kind not in {"number", "integer"}:
            # Adjacent words are often incorrectly separated into layout cells.
            # Include complete phrases and suffixes without asking Jev to generate.
            words = row["text"].split()
            spans = (
                [" ".join(words[i:j]) for i in range(len(words)) for j in range(i + 1, min(len(words), i + 8) + 1)]
                if len(words) < 35
                else []
            )
            for span in spans:
                if len(span) > 160:
                    continue
                overlap = len(query & fields._tokens(span)) / max(1, len(query))
                expanded.append(
                    (
                        label_score * 3 + near_score + desc_score - overlap * 0.6 + min(len(span), 40) / 400,
                        {
                            "value": span,
                            "context": f"SOURCE ROW: {row['raw']}\nNearby: {row['context']}",
                            "page": row["page"],
                        },
                    )
                )
    expanded.sort(key=lambda pair: pair[0], reverse=True)
    # Round-robin across rows prevents n-grams from one long label starving other
    # locations. Reserve exact whole/cell candidates independently of n-grams.
    seed = fields.candidates(lines, path, node, limit=32)
    results, seen, per_context = [], set(), {}
    for option in seed:
        key = json.dumps(option["value"], ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            results.append(option)
    for _, option in expanded:
        key = json.dumps(option["value"], ensure_ascii=False)
        context = option["context"]
        if key in seen or per_context.get(context, 0) >= 14:
            continue
        seen.add(key)
        per_context[context] = per_context.get(context, 0) + 1
        results.append(option)
        if len(results) >= limit:
            break
    return results


def _relevant(lines: list[dict], path: str, node: dict) -> list[dict]:
    query = fields._tokens(path + " " + node.get("description", ""))
    return sorted(lines, key=lambda line: len(query & fields._tokens(line["context"])), reverse=True)


def extract(document: dict, schema: dict, client) -> dict:
    schema = resolve_refs(schema)
    lines = geometric_lines(document)
    holder, jobs, arrays = {}, [], []

    def walk(node, container, key, path):
        node = _effective(node)
        kind = fields._kind(node)
        if kind == "object":
            container[key] = {}
            for name, child in node.get("properties", {}).items():
                walk(child, container[key], name, f"{path}.{name}".strip("."))
        elif kind == "array":
            container[key] = []
            arrays.append((path, node, container, key))
        else:
            container[key] = None
            jobs.append((path, node, container, key))

    walk(schema, holder, "data", "")
    questions, options_by_id = {}, {}
    for i, (path, node, _, _) in enumerate(jobs):
        options = span_candidates(lines, path, node)
        options_by_id[f"g{i}"] = options
        local = "\n".join(line["raw"] for line in _relevant(lines, path, node)[:16])[:4200]
        questions[f"g{i}"] = fields._question(
            f"Extract field {path}. Schema: {json.dumps(node)}.\nLOCAL SOURCE:\n{local}\n"
            "Choose the complete VALUE, never its label, form box number, instruction or unrelated subtotal. "
            "Use physical source rows for number-label association. Names/addresses may span several words. "
            "Choose none when absent, blank or unsupported; "
            "a printed boolean question alone does not establish true/false. "
            "Do not infer a missing value from common knowledge.",
            options,
        )
    answers = (
        client.decide("Extract values from the supplied localized document evidence.", questions) if questions else {}
    )
    evidence = []
    for i, (path, _node, container, key) in enumerate(jobs):
        answer = answers.get(f"g{i}", {})
        selected = fields._selected(answer, options_by_id[f"g{i}"])
        if selected is not None:
            container[key] = selected["value"]
            evidence.append(
                {
                    "path": path,
                    "page_index": selected["page"],
                    "text": selected["context"],
                    "confidence": answer.get("confidence"),
                }
            )
    # Table mapping processes every row in the selected table, with no row cap.
    array_document = {"text": "\n".join(row["raw"] for row in lines)}
    for path, node, container, key in arrays:
        result = tables.extract(array_document, {"type": "object", "properties": {key: node}}, client)
        container[key] = result["data"][key]
        for item in result.get("evidence", []):
            item = dict(item)
            item["path"] = path + item["path"][len(str(key)) :]
            evidence.append(item)
    return {"data": holder["data"], "evidence": evidence}
