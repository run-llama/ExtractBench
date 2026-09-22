"""One Jev decision per field over a union of deterministic source candidates.

Consensus refers to candidate generators, not multiple model votes. The model
never generates text, and no per-candidate evidence is duplicated in the prompt.
"""

from __future__ import annotations

import json
import math
from itertools import zip_longest

from ..table_codegen.schema_utils import _effective, resolve_refs
from . import variant_fields as fields
from .variant_boundaries import _array_schema, _merge
from .variant_geometric import geometric_lines
from .variant_localized import _normalize, _regions, _retrieve, _values


def candidates(document, path, node, lines=None, geometry=None, regions=None):
    lines = fields._lines(document) if lines is None else lines
    geometry = geometric_lines(document) if geometry is None else geometry
    regions = _regions(document) if regions is None else regions
    local = _retrieve(regions, path, node, 4)
    kind = fields._kind(node)
    if "enum" in node:
        choices = [{"value": value, "page": None, "context": ""} for value in node["enum"] if value is not None]
    elif kind == "boolean":
        choices = [{"value": value, "page": None, "context": ""} for value in (True, False)]
    else:
        # Preserve a fixed allowance for each independent generator, then fill
        # remaining capacity by interleaving localized regions rather than
        # letting the first region consume every candidate slot.
        choices = fields.candidates(lines, path, node, limit=60)
        choices += fields.candidates(geometry, path, node, limit=40)
        region_choices = []
        for region in local:
            values = []
            for raw in _values(region["text"], node):
                value = _normalize(raw, node) if kind in ("number", "integer") else raw
                if value is None or isinstance(value, float) and not math.isfinite(value):
                    continue
                values.append(
                    {"value": value, "page": region["page_index"], "context": region["text"], "source_text": raw}
                )
            region_choices.append(values)
        choices += [item for batch in zip_longest(*region_choices) for item in batch if item is not None]
    unique, seen = [], set()
    for item in choices:
        key = json.dumps(item["value"], ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
        if len(unique) >= 200:
            break
    # Context includes the top independent candidate neighborhoods even if
    # their lexical rank differs from the localized region shortlist.
    contexts, used = [], set()
    seeds = [f"Page {r['page_index'] + 1}:\n{r['text']}" for r in local[:3]]
    seeds += [item["context"] for item in unique[:8] if item.get("context")]
    size = 0
    for context in seeds:
        if context in used:
            continue
        used.add(context)
        if size + len(context) > 6500:
            continue
        contexts.append(context)
        size += len(context)
    return unique, "\n\n".join(contexts)


def extract(document: dict, schema: dict, client) -> dict:
    schema = resolve_refs(schema)
    lines, geometry, regions = fields._lines(document), geometric_lines(document), _regions(document)
    jobs, holder, evidence = [], {}, []

    def walk(node, path, dest, key):
        node = _effective(node)
        kind = fields._kind(node)
        if kind == "object":
            dest[key] = {}
            for name, child in node.get("properties", {}).items():
                walk(child, f"{path}.{name}".strip("."), dest[key], name)
        elif kind == "array":
            dest[key] = []
        else:
            dest[key] = None
            jobs.append((path, node, dest, key))

    walk(schema, "", holder, "data")
    for start in range(0, len(jobs), 12):
        batch = jobs[start : start + 12]
        questions, options = {}, {}
        for i, (path, node, _, _) in enumerate(batch):
            values, context = candidates(document, path, node, lines, geometry, regions)
            options[str(i)] = values
            questions[str(i)] = {
                "type": "choice",
                "instructions": f"Extract {path}. Field description: {node.get('description', '')}. Type: {fields._kind(node)}.\nSOURCE:\n{context}\nChoose the complete actual field value, excluding its label and neighboring values. Schema examples are not evidence. Match the correct named entity. For checkboxes, a printed label is not a mark; honor explicit false/null rules. Choose absent if blank or unsupported.",  # noqa: E501
                "criteria": {
                    "none": "Absent, blank, or not supported by the source",
                    **{f"c{j}": json.dumps(item["value"], ensure_ascii=False) for j, item in enumerate(values)},
                },
            }
        answers = (
            client.decide(
                "Resolve each extraction field from its local source and deterministic candidate union.", questions
            )
            if questions
            else {}
        )
        for i, (path, _node, dest, key) in enumerate(batch):
            selected = fields._selected(answers.get(str(i), {}), options[str(i)])
            if selected is not None:
                dest[key] = selected["value"]
                evidence.append(
                    {
                        "path": path,
                        "page_index": selected.get("page"),
                        "source_text": selected.get("source_text", selected["value"]),
                        "context": selected.get("context", ""),
                    }
                )
    # Same table engine as localized: isolate the scalar candidate-union change.
    array_schema = _array_schema(schema, schema)
    if array_schema is not None:
        from .variant_tables import extract as extract_arrays

        result = extract_arrays(document, array_schema, client)
        if isinstance(holder["data"], dict):
            _merge(holder["data"], result["data"])
        else:
            holder["data"] = result["data"]
        evidence.extend(result.get("evidence", []))
    return {"data": holder["data"], "evidence": evidence}
