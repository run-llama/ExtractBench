"""Route long repeated Markdown tables to anchors and other sources to compact.

Routing reads only parsed source structure and the public extraction schema.
The fixed 100-row threshold protects full-row coverage on long repeated tables
while retaining compact extraction for ordinary documents.
"""

from __future__ import annotations

import re

from . import variant_anchors, variant_compact

TABLE_ROW_THRESHOLD = 100


def markdown_row_count(document: dict) -> int:
    markdown = document.get("markdown") or "\n".join(page.get("markdown") or "" for page in document.get("pages", []))
    count = 0
    for line in markdown.splitlines():
        if not re.search(r"(?<!\\)\|", line):
            continue
        cells = re.split(r"(?<!\\)\|", line.strip())
        if cells and not cells[0].strip():
            cells = cells[1:]
        if cells and not cells[-1].strip():
            cells = cells[:-1]
        cells = [cell.strip() for cell in cells]
        if not cells or not any(cells):
            continue
        if all(not cell or re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        count += 1
    return count


def has_object_array(schema: dict) -> bool:
    """Inspect reachable properties and unions; resolve local refs cycle-safely."""

    def dereference(node, visited):
        reference = node.get("$ref")
        if not reference or reference in visited or not reference.startswith("#/"):
            return node, visited
        target = schema
        for segment in reference[2:].split("/"):
            if not isinstance(target, dict):
                return node, visited
            target = target.get(segment.replace("~1", "/").replace("~0", "~"), {})
        if not isinstance(target, dict):
            return node, visited
        return dereference(
            {**target, **{key: value for key, value in node.items() if key != "$ref"}}, visited | {reference}
        )

    def branches(node):
        return [
            child
            for keyword in ("anyOf", "oneOf", "allOf")
            for child in node.get(keyword, [])
            if isinstance(child, dict)
        ]

    def is_object(node, visited):
        node, visited = dereference(node, visited)
        kind = node.get("type")
        return (
            kind == "object"
            or isinstance(kind, list)
            and "object" in kind
            or "properties" in node
            or any(is_object(child, visited) for child in branches(node))
        )

    def walk(node, visited, depth=0):
        if not isinstance(node, dict) or depth > 40:
            return False
        node, visited = dereference(node, visited)
        items = node.get("items")
        kind = node.get("type")
        if isinstance(items, dict) and (kind == "array" or isinstance(kind, list) and "array" in kind or kind is None):
            if is_object(items, visited):
                return True
        children = list(node.get("properties", {}).values()) + branches(node)
        if isinstance(items, dict):
            children.append(items)
        return any(walk(child, visited, depth + 1) for child in children)

    return walk(schema, set())


def extract(document: dict, schema: dict, client) -> dict:
    rows = markdown_row_count(document)
    object_array = has_object_array(schema)
    route = "anchors" if rows >= TABLE_ROW_THRESHOLD and object_array else "compact"
    engine = variant_anchors if route == "anchors" else variant_compact
    prediction = engine.extract(document, schema, client)
    return {
        **prediction,
        "route_diagnostics": {
            "engine": route,
            "markdown_rows": rows,
            "has_object_array": object_array,
            "table_row_threshold": TABLE_ROW_THRESHOLD,
        },
    }
