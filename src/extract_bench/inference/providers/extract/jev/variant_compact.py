"""Localized extraction with compact scalar questions and identical candidates.

The underlying source retrieval, candidate enumeration, array extraction, and
normalization are unchanged. Only scalar question instructions are rewritten.
"""

from __future__ import annotations

import json
import re

from . import variant_localized

_REGION_PREFIX = "Find the source region containing the VALUE (not merely a mention) of "
_VALUE_SUFFIX = (
    "\nChoose only the actual field value, excluding labels, adjacent columns, line numbers, and explanatory examples. "
    "Honor any explicit blank/false/null rules. Do not treat an unmarked checkbox as checked."
)


def _field_details(path: str, schema: dict) -> str:
    """Keep field semantics while removing JSON syntax and irrelevant metadata."""
    title = schema.get("title")
    label = path + (f" ({title})" if title and str(title).lower() != path.lower() else "")
    parts = [label]
    description = schema.get("description")
    if description:
        # Do not truncate descriptions: their tails often specify null behavior,
        # measurement units, scaling, or normalized output formats.
        parts.append(re.sub(r"\s+", " ", str(description)).strip())
    constraints = {
        key: value
        for key, value in schema.items()
        if key
        in {
            "type",
            "format",
            "pattern",
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "multipleOf",
            "minLength",
            "maxLength",
            "const",
            "default",
            "not",
            "anyOf",
            "oneOf",
            "allOf",
            "if",
            "then",
            "else",
            "dependentSchemas",
        }
    }
    if constraints:
        parts.append("Output rules: " + json.dumps(constraints, ensure_ascii=False, separators=(",", ":")))
    return ". ".join(parts)


def compact_instructions(instructions: str) -> str:
    if instructions.startswith(_REGION_PREFIX):
        prefix, region = _REGION_PREFIX, True
    elif instructions.startswith("Extract ") and ". Field schema: " in instructions:
        prefix, region = "Extract ", False
    else:
        # In particular, the table engine's questions pass through unchanged.
        return instructions
    try:
        path, remaining = instructions[len(prefix) :].split(". Field schema: ", 1)
        schema, end = json.JSONDecoder().raw_decode(remaining)
    except (ValueError, TypeError):
        return instructions
    if not isinstance(schema, dict):
        return instructions
    details = _field_details(path, schema)
    if region:
        return f"Where is the value of {details}? Blank checkboxes are relevant evidence."
    suffix = remaining[end:]
    if not suffix.startswith("\nSOURCE:\n") or not suffix.endswith(_VALUE_SUFFIX):
        # Avoid accidentally dropping source text if the underlying prompt changes.
        return instructions
    source = suffix[len("\nSOURCE:\n") : -len(_VALUE_SUFFIX)]
    return (
        f"What is {details}?\nSOURCE:\n{source}\n"
        "Select its value, not its label. If absent choose none; apply explicit blank/false rules."
    )


class CompactClient:
    def __init__(self, client):
        self.client = client

    def decide(self, state, questions):
        compact = {
            identifier: {
                **question,
                "instructions": compact_instructions(question.get("instructions", "")),
            }
            for identifier, question in questions.items()
        }
        return self.client.decide(state, compact)


def extract(document: dict, schema: dict, client) -> dict:
    return variant_localized.extract(document, schema, CompactClient(client))
