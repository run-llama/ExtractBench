from datetime import UTC, datetime

import pytest

from extract_bench.inference.providers.base import ProviderPermanentError
from extract_bench.inference.providers.extract.nuextract3_extract import (
    NuExtract3ExtractProvider,
    _json_schema_to_nuextract_template_and_instructions,
)
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import InferenceRequest, RawInferenceResult
from extract_bench.schemas.product import ProductType


def test_nullable_ref_keeps_row_template_and_descriptions() -> None:
    schema = {
        "$defs": {
            "item": {
                "type": "object",
                "description": "One line item.",
                "properties": {"quantity": {"type": "integer", "description": "Units ordered."}},
            },
        },
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "anyOf": [{"$ref": "#/$defs/item"}, {"type": "null"}],
                    "description": "Include every item.",
                },
            },
        },
    }

    template, instructions = _json_schema_to_nuextract_template_and_instructions(schema)

    assert template == {"items": [{"quantity": "integer"}]}
    assert instructions.splitlines() == [
        "$.items[]: Include every item.",
        "$.items[]: One line item.",
        "$.items[].quantity: Units ordered.",
    ]


def test_allof_keeps_fields_from_each_branch() -> None:
    schema = {
        "allOf": [
            {"type": "object", "properties": {"id": {"type": "integer"}}},
            {"type": "object", "properties": {"label": {"type": "string"}}},
        ],
    }

    template, _ = _json_schema_to_nuextract_template_and_instructions(schema)

    assert template == {"id": "integer", "label": "string"}


def test_string_annotations_and_nullable_enums_use_supported_template_leaves() -> None:
    schema = {
        "type": "object",
        "properties": {
            "quote": {"type": "string", "x-verbatim": True},
            "issued": {"type": "string", "description": "date"},
            "contact": {"type": "string", "format": "email"},
            "status": {"type": ["string", "null"], "enum": ["Paid", None]},
        },
    }

    template, _ = _json_schema_to_nuextract_template_and_instructions(schema)

    assert template == {
        "quote": "verbatim-string",
        "issued": "date",
        "contact": "email-address",
        "status": "string",
    }


@pytest.mark.parametrize(
    ("schema", "message"),
    [
        ({"anyOf": [{"type": "string"}, {"type": "integer"}]}, "Ambiguous unions"),
        ({"$defs": {"node": {"$ref": "#/$defs/node"}}, "$ref": "#/$defs/node"}, "Cyclic"),
    ],
    ids=["ambiguous-union", "cyclic-ref"],
)
def test_unrepresentable_schema_fails_permanently(schema: dict[str, object], message: str) -> None:
    with pytest.raises(ProviderPermanentError, match=message):
        _json_schema_to_nuextract_template_and_instructions(schema)


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (
            '{"total":3,"items":[{"sku":"A"},{"sku":"B"},{"sku":"unfinish',
            {"total": 3, "items": [{"sku": "A"}, {"sku": "B"}]},
        ),
        ('<think>Inspect the document.</think>\n```json\n{"total":3}\n```', {"total": 3}),
        ("I cannot extract this document.", {}),
    ],
    ids=["truncated-array", "fenced-json", "refusal"],
)
def test_normalize_scores_the_recoverable_json(content: str, expected: dict[str, object]) -> None:
    provider = NuExtract3ExtractProvider("nuextract3_extract", {"server_url": "http://unused"})
    pipeline = PipelineSpec(
        pipeline_name="nuextract3_extract",
        provider_name="nuextract3_extract",
        product_type=ProductType.EXTRACT,
    )
    now = datetime.now(UTC)
    raw = RawInferenceResult(
        request=InferenceRequest(
            example_id="document",
            source_file_path="document.pdf",
            product_type=ProductType.EXTRACT,
        ),
        pipeline=pipeline,
        pipeline_name=pipeline.pipeline_name,
        product_type=ProductType.EXTRACT,
        raw_output={"content": content},
        started_at=now,
        completed_at=now,
        latency_in_ms=0,
    )

    result = provider.normalize(raw)

    assert result.output.extracted_data == expected
