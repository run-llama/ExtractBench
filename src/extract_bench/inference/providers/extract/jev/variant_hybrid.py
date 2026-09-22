"""Source-size routing for scalars, complete multipage column extraction for arrays.

The route depends only on PDF-derived page/text size. No document identifiers,
benchmark labels, or domain-specific schemas enter routing or normalization.
"""

from __future__ import annotations

import re

from ..table_codegen.schema_utils import _effective, resolve_refs
from . import variant_fields, variant_localized, variant_multipage


def _false_when_unmarked(node: dict) -> bool:
    description = str(node.get("description", "")).lower()
    return node.get("default") is False or any(
        re.search(pattern, description)
        for pattern in (
            r"false\s+otherwise",
            r"otherwise\s*[,;:]?\s*(?:return\s+)?false",
            r"(?:blank|unchecked|unmarked|not checked)[^.]{0,90}\bfalse\b",
            r"\bfalse\b[^.]{0,90}(?:blank|unchecked|unmarked|not checked)",
            r"never return null[^.]{0,90}\bfalse\b",
        )
    )


def split_schema(schema: dict) -> tuple[dict | None, dict | None]:
    """Separate scalar and array branches without changing their original paths."""
    schema = _effective(schema)
    kind = variant_fields._kind(schema)
    if kind == "array":
        return None, schema
    if kind != "object":
        if kind == "boolean":
            schema = dict(schema)
            schema["description"] = schema.get("description", "") + (
                " A printed checkbox label alone is not a checked mark. "
                "Return true only for an explicit affirmative or checked mark. "
                "Follow the schema's explicit blank/false/null rule."
            )
        return schema, None
    scalars, arrays = {}, {}
    for name, child in schema.get("properties", {}).items():
        scalar, array = split_schema(child)
        if scalar is not None:
            scalars[name] = scalar
        if array is not None:
            arrays[name] = array

    def branch(properties):
        if not properties:
            return None
        result = {**schema, "properties": properties}
        if "required" in result:
            result["required"] = [name for name in result["required"] if name in properties]
        return result

    return branch(scalars), branch(arrays)


def _merge(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        result = dict(left)
        for key, value in right.items():
            result[key] = _merge(result[key], value) if key in result else value
        return result
    return right if right is not None else left


def normalize(data, schema: dict):
    schema = _effective(schema)
    kind = variant_fields._kind(schema)
    if kind == "object":
        source = data if isinstance(data, dict) else {}
        return {name: normalize(source.get(name), child) for name, child in schema.get("properties", {}).items()}
    if kind == "array":
        return [normalize(item, schema.get("items", {})) for item in data] if isinstance(data, list) else []
    if kind == "boolean" and data is None and _false_when_unmarked(schema):
        return False
    if isinstance(data, str):
        return re.sub(r"\s+", " ", data).strip()
    return data


def scalar_route(document: dict) -> str:
    text = document.get("text") or "\n".join(page.get("text", "") for page in document.get("pages", []))
    # Whole-document context is useful for sparse short forms. Large sources
    # need lexical retrieval so later pages are not lost to a prompt window.
    return "fields" if len(text) <= 14000 and len(document.get("pages", [])) <= 5 else "localized"


def extract(document: dict, schema: dict, client) -> dict:
    schema = resolve_refs(schema)
    scalar_schema, array_schema = split_schema(schema)
    data, evidence = None, []
    if scalar_schema is not None:
        engine = variant_fields if scalar_route(document) == "fields" else variant_localized
        result = engine.extract(document, scalar_schema, client)
        data = result["data"]
        evidence.extend(result.get("evidence", []))
    if array_schema is not None:
        # This engine visits every region/page and every source row. The hybrid
        # adds no page, region, or output record ceilings.
        result = variant_multipage.extract(document, array_schema, client)
        data = _merge(data, result["data"])
        evidence.extend(result.get("evidence", []))
    return {"data": normalize(data, schema), "evidence": evidence}
