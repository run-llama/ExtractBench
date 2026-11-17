from __future__ import annotations

import re
from typing import Any


def path_tokens(path: str) -> list[str | int]:
    tokens: list[str | int] = []
    for part in path.split("."):
        if not part:
            continue
        match = re.match(r"^([^\[]+)", part)
        if match:
            tokens.append(match.group(1))
        for index in re.findall(r"\[(\d+)\]", part):
            tokens.append(int(index))
    return tokens


def iter_leaf_values(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        leaves: list[tuple[str, Any]] = []
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                continue
            child_prefix = f"{prefix}.{key}" if prefix else key
            leaves.extend(iter_leaf_values(child, child_prefix))
        return leaves
    if isinstance(value, list):
        leaves = []
        for index, child in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            leaves.extend(iter_leaf_values(child, child_prefix))
        return leaves
    return [(prefix, value)] if prefix else []


def resolve_schema_ref(root: Any, node: Any) -> Any:
    seen: set[str] = set()
    while isinstance(node, dict) and isinstance(node.get("$ref"), str):
        ref = node["$ref"]
        if ref in seen or not ref.startswith("#/"):
            return node
        seen.add(ref)
        cursor = root
        for part in ref[2:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            if not isinstance(cursor, dict) or part not in cursor:
                return node
            cursor = cursor[part]
        node = cursor
    return node


def concrete_schema_node(root: Any, node: Any) -> Any:
    node = resolve_schema_ref(root, node)
    while isinstance(node, dict):
        variants = node.get("anyOf") or node.get("oneOf")
        if not isinstance(variants, list):
            break
        candidates = [
            resolve_schema_ref(root, candidate)
            for candidate in variants
            if not (isinstance(candidate, dict) and candidate.get("type") == "null")
        ]
        if not candidates:
            break
        preferred = None
        for candidate in candidates:
            candidate = resolve_schema_ref(root, candidate)
            if isinstance(candidate, dict) and (
                candidate.get("type") in {"object", "array"} or "properties" in candidate or "items" in candidate
            ):
                preferred = candidate
                break
        node = preferred if preferred is not None else candidates[0]
        node = resolve_schema_ref(root, node)
    return node


def schema_at_path(schema: Any, tokens: list[str | int]) -> Any:
    cursor = schema
    for token in tokens:
        cursor = concrete_schema_node(schema, cursor)
        if not isinstance(cursor, dict):
            return None
        if isinstance(token, int):
            cursor = concrete_schema_node(schema, cursor.get("items"))
            continue
        if cursor.get("type") == "array" or ("items" in cursor and "properties" not in cursor):
            cursor = concrete_schema_node(schema, cursor.get("items"))
            if not isinstance(cursor, dict):
                return None
        props = cursor.get("properties")
        if not isinstance(props, dict) or token not in props:
            return None
        cursor = props[token]
    return concrete_schema_node(schema, cursor)


def schema_path_exists(schema: Any, path: str) -> bool:
    return schema_at_path(schema, path_tokens(path)) is not None


def path_without_indices(path: str) -> str:
    return re.sub(r"\[\d+\]", "[]", path)
