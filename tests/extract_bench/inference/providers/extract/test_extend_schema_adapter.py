"""Tests for Extend schema adaptation."""

import pytest

pytest.importorskip("extend_ai", reason="dev and runners extras required; run: uv sync --extra dev --extra runners")

from extract_bench.inference.providers.extract.extend import _adapt_schema_for_extend


def test_extend_schema_adapter_preserves_nullable_scalar_unions() -> None:
    schema = {
        "type": "object",
        "required": ["contract_number"],
        "properties": {
            "contract_number": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "description": "Contract number.",
                "default": None,
                "title": "Contract Number",
            },
            "page_count": {
                "type": ["integer", "null"],
                "description": "Nullable integer encoded with a JSON Schema type list.",
            },
        },
    }

    adapted, primitive_array_paths, _renamed = _adapt_schema_for_extend(schema)

    assert adapted["required"] == ["contract_number"]
    assert primitive_array_paths == {}
    assert adapted["properties"]["contract_number"] == {
        "type": ["string", "null"],
        "description": "Contract number.",
    }
    assert adapted["properties"]["page_count"] == {
        "type": ["integer", "null"],
        "description": "Nullable integer encoded with a JSON Schema type list.",
    }


def test_extend_schema_adapter_keeps_nullable_containers_optional() -> None:
    schema = {
        "type": "object",
        "required": ["codes", "line_items"],
        "properties": {
            "codes": {
                "anyOf": [
                    {"type": "array", "items": {"type": "string"}},
                    {"type": "null"},
                ],
                "description": "Optional primitive array.",
            },
            "line_items": {
                "anyOf": [
                    {
                        "type": "array",
                        "items": {"$ref": "#/$defs/LineItem"},
                    },
                    {"type": "null"},
                ],
            },
        },
        "$defs": {
            "LineItem": {
                "type": "object",
                "title": "LineItem",
                "properties": {
                    "description": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "default": None,
                        "title": "Description",
                    }
                },
            }
        },
    }

    adapted, primitive_array_paths, _renamed = _adapt_schema_for_extend(schema)

    assert adapted["required"] == []
    assert primitive_array_paths == {"codes": ["string"]}
    assert adapted["properties"]["codes"] == {
        "type": "array",
        "items": {"type": "object", "properties": {"value": {"type": "string"}}},
        "description": "Optional primitive array.",
    }
    assert adapted["properties"]["line_items"] == {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "description": {
                    "type": ["string", "null"],
                }
            },
        },
    }
    assert "$defs" not in adapted


def test_extend_schema_adapter_resolves_nullable_ref_unions() -> None:
    schema = {
        "type": "object",
        "properties": {
            "ultimate_consignee": {
                "anyOf": [{"$ref": "#/$defs/Party"}, {"type": "null"}],
                "default": None,
                "description": "Optional party object.",
                "title": "Ultimate Consignee",
            },
        },
        "$defs": {
            "Party": {
                "type": "object",
                "title": "Party",
                "properties": {
                    "name": {
                        "type": ["string", "null"],
                        "default": None,
                        "title": "Name",
                    }
                },
            }
        },
    }

    adapted, primitive_array_paths, _renamed = _adapt_schema_for_extend(schema)

    assert primitive_array_paths == {}
    assert adapted["properties"]["ultimate_consignee"] == {
        "type": "object",
        "properties": {
            "name": {
                "type": ["string", "null"],
            }
        },
        "description": "Optional party object.",
    }


def test_extend_schema_adapter_renames_reserved_id_key_and_restores_it() -> None:
    from extract_bench.inference.providers.extract.extend import _adapt_result_from_extend

    schema = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Observation ID."},
                        "note": {"type": "string"},
                    },
                },
            }
        },
    }

    adapted, primitive_array_paths, renamed = _adapt_schema_for_extend(schema)

    items_props = adapted["properties"]["rows"]["items"]["properties"]
    # 'id' must not survive in the schema Extend sees, but a sibling stays put.
    assert "id" not in items_props
    assert "id_field" in items_props
    assert "note" in items_props
    assert renamed == {"rows[items]": {"id_field": "id"}}

    # A result keyed by the alias round-trips back to the original 'id' key.
    extend_result = {"rows": [{"id_field": "OBS-1", "note": "x"}, {"id_field": "OBS-2", "note": "y"}]}
    restored = _adapt_result_from_extend(extend_result, primitive_array_paths, renamed)
    assert restored == {"rows": [{"id": "OBS-1", "note": "x"}, {"id": "OBS-2", "note": "y"}]}


def test_extend_schema_adapter_wraps_empty_object_array_items() -> None:
    from extract_bench.inference.providers.extract.extend import _adapt_result_from_extend

    schema = {
        "type": "object",
        "properties": {
            "currency_code": {
                "type": "array",
                "description": "This profile has no fact of this type; return an empty array.",
                "items": {"type": "object", "additionalProperties": False, "properties": {}},
            }
        },
    }

    adapted, primitive_array_paths, _renamed = _adapt_schema_for_extend(schema)

    # Empty-object items become a wrapped primitive so Extend accepts the schema.
    assert primitive_array_paths == {"currency_code": ["string"]}
    assert adapted["properties"]["currency_code"]["items"] == {
        "type": "object",
        "properties": {"value": {"type": "string"}},
    }
    # The ground truth is always [] -> round-trips unchanged.
    assert _adapt_result_from_extend({"currency_code": []}, primitive_array_paths) == {"currency_code": []}


def test_extend_schema_adapter_strips_rejected_examples_without_removing_named_fields() -> None:
    field = {"type": "string", "description": "Shareholder name.", "examples": ["Jane Doe"]}
    schema = {
        "type": "object",
        "properties": {
            "form_7203": {
                "type": "object",
                "properties": {"name_of_shareholder": field, "examples": {"type": "string"}},
            },
            "rows": {"type": "array", "items": {"type": "object", "properties": {"name": field}}},
        },
    }

    adapted, _, _ = _adapt_schema_for_extend(schema)

    form_properties = adapted["properties"]["form_7203"]["properties"]
    expected_field = {"type": "string", "description": "Shareholder name."}
    assert form_properties["name_of_shareholder"] == expected_field
    assert adapted["properties"]["rows"]["items"]["properties"]["name"] == expected_field
    assert form_properties["examples"] == {"type": "string"}
    assert field["examples"] == ["Jane Doe"]
