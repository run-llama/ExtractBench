"""Offline tests for the Parseur provider: schema conversion, result decoding, HTTP flow."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import requests

pytest.importorskip("pypdf", reason="dev and runners extras required; run: uv sync --extra dev --extra runners")

from pypdf import PdfWriter

from extract_bench.inference.pipelines import get_pipeline
from extract_bench.inference.providers.base import (
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from extract_bench.inference.providers.extract.parseur import (
    API_BASE_URL,
    BOOLEAN_CHOICES,
    MAX_EMPTY_RESULT_POLLS,
    MAX_POLL_ERRORS,
    ParseurProvider,
    conversion_hash,
    convert_json_schema,
    decode_result,
)
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import InferenceRequest
from extract_bench.schemas.product import ProductType

SCHEMA: dict[str, Any] = {
    "$defs": {
        "Address": {
            "type": "object",
            "properties": {
                "street": {"type": "string", "description": "Street line"},
                "zip": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
        },
        "LineItem": {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "quantity": {"anyOf": [{"type": "integer"}, {"type": "null"}], "description": "Units"},
                "unit_price": {"type": ["number", "null"]},
                "taxable": {"type": "boolean"},
                "dimensions": {
                    "type": "object",
                    "properties": {"width": {"type": "number"}, "height": {"type": "number"}},
                },
                "discounts": {"type": "array", "items": {"$ref": "#/$defs/Discount"}},
            },
        },
        "Discount": {"type": "object", "properties": {"code": {"type": "string"}}},
    },
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string", "description": "Verbatim as printed"},
        "invoice_date": {"anyOf": [{"type": "string", "format": "date"}, {"type": "null"}]},
        "total": {"type": "number"},
        "page_count": {"type": "integer"},
        "paid": {"type": "boolean"},
        "currency": {"type": "string", "enum": ["USD", "EUR"]},
        "priority": {"type": "integer", "enum": [1, 2, 3]},
        "customer": {
            "type": "object",
            "properties": {"id": {"type": "string"}, "address": {"$ref": "#/$defs/Address"}},
        },
        "tags": {"type": "array", "items": {"type": "string"}, "description": "Free tags"},
        "line_items": {"type": "array", "items": {"$ref": "#/$defs/LineItem"}},
        "matrix": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
    },
}

# Levels: `tables` 1, `tables.rows` and `tables.column_headers` 2, `tables.rows.values` 3.
DEEP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tables": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "table_number": {"type": "string"},
                    "column_headers": {"type": "array", "items": {"type": "string"}},
                    "rows": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "row_label": {"type": "string"},
                                "values": {"type": "array", "items": {"type": "string"}},
                            },
                        },
                    },
                },
            },
        }
    },
}


def _by_name(parser_object_set: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {field["name"]: field for field in parser_object_set}


# ----------------------------------------------------------------------------
# JSON Schema -> Parseur field tree
# ----------------------------------------------------------------------------


def test_convert_maps_scalars_to_parseur_formats() -> None:
    conversion = convert_json_schema(SCHEMA)
    fields = _by_name(conversion["parser_object_set"])

    assert fields["invoice_number"] == {"name": "invoice_number", "format": "ONELINE", "query": "Verbatim as printed"}
    assert fields["invoice_date"]["format"] == "DATE"
    assert fields["total"]["format"] == "NUMBER"
    assert fields["page_count"]["format"] == "NUMBER"
    assert fields["paid"]["format"] == "ONELINE"
    assert fields["paid"]["choice_set"] == BOOLEAN_CHOICES
    assert "true" in fields["paid"]["query"]
    assert fields["currency"] == {"name": "currency", "format": "ONELINE", "choice_set": ["EUR", "USD"]}
    assert fields["priority"]["choice_set"] == ["1", "2", "3"]
    # Nothing is marked required: the bench wants null for absent values, not a guess.
    assert all("is_required" not in field for field in conversion["parser_object_set"])


def test_convert_flattens_nested_objects_with_dotted_names() -> None:
    conversion = convert_json_schema(SCHEMA)
    fields = _by_name(conversion["parser_object_set"])

    assert fields["customer.id"]["format"] == "ONELINE"
    assert fields["customer.address.street"] == {
        "name": "customer.address.street",
        "format": "ONELINE",
        "query": "Street line",
    }
    assert fields["customer.address.zip"]["format"] == "ONELINE"
    assert conversion["fields"]["customer.address.zip"]["path"] == ["customer", "address", "zip"]


def test_convert_arrays_become_tables() -> None:
    conversion = convert_json_schema(SCHEMA)
    fields = _by_name(conversion["parser_object_set"])

    tags = fields["tags"]
    assert tags["format"] == "TABLE"
    assert tags["query"] == "Free tags"
    assert [c["name"] for c in tags["parser_object_set"]] == ["value"]
    assert conversion["fields"]["tags"]["kind"] == "scalar_array"

    items = fields["line_items"]
    assert items["format"] == "TABLE"
    columns = _by_name(items["parser_object_set"])
    assert columns["description"]["format"] == "ONELINE"
    assert columns["quantity"] == {"name": "quantity", "format": "NUMBER", "query": "Units"}
    assert columns["unit_price"]["format"] == "NUMBER"
    assert columns["taxable"]["choice_set"] == BOOLEAN_CHOICES
    # Object inside a row -> dotted column names.
    assert columns["dimensions.width"]["format"] == "NUMBER"
    assert conversion["fields"]["line_items"]["columns"]["dimensions.width"]["path"] == ["dimensions", "width"]
    # Table inside a row -> TABLE column.
    assert columns["discounts"]["format"] == "TABLE"
    assert [c["name"] for c in columns["discounts"]["parser_object_set"]] == ["code"]
    assert conversion["fields"]["line_items"]["columns"]["discounts"]["kind"] == "table"
    # Array of arrays has no Parseur shape.
    assert "matrix" not in fields


@pytest.mark.parametrize(
    ("schema", "expected_dropped"),
    [(SCHEMA, ["matrix"]), (DEEP_SCHEMA, [])],
)
def test_shapes_parseur_cannot_hold_are_dropped(schema: dict[str, Any], expected_dropped: list[str]) -> None:
    """An array of arrays has no Parseur shape: it is dropped with a warning."""
    assert convert_json_schema(schema)["dropped"] == expected_dropped


def test_all_of_branches_merge_their_properties() -> None:
    schema = {
        "type": "object",
        "properties": {
            "party": {
                "allOf": [
                    {"type": "object", "properties": {"name": {"type": "string"}}},
                    {"type": "object", "properties": {"amount": {"type": "number"}}},
                ]
            }
        },
    }
    fields = _by_name(convert_json_schema(schema)["parser_object_set"])
    assert fields["party.name"]["format"] == "ONELINE"
    assert fields["party.amount"]["format"] == "NUMBER"


def test_recursive_schemas_convert() -> None:
    """A self-referential ``$ref``, through an array or an object, stops instead of recursing without end."""
    node = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "parent": {"$ref": "#/$defs/Node"},
            "children": {"type": "array", "items": {"$ref": "#/$defs/Node"}},
        },
    }
    schema = {"$defs": {"Node": node}, "type": "object", "properties": {"root": {"$ref": "#/$defs/Node"}}}
    fields = _by_name(convert_json_schema(schema)["parser_object_set"])
    assert {"root.name", "root.parent", "root.children"} <= set(fields)


def test_object_descriptions_go_to_their_fields() -> None:
    """Parseur has no object field: an object's description is carried by each of its fields."""
    schema = {
        "type": "object",
        "properties": {
            "buyer": {
                "type": "object",
                "description": "The party that pays",
                "properties": {
                    "name": {"type": "string", "description": "Legal name"},
                    "address": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        },
    }
    fields = _by_name(convert_json_schema(schema)["parser_object_set"])
    assert fields["buyer.name"]["query"] == "[buyer: The party that pays] Legal name"
    assert fields["buyer.address.city"]["query"] == "[address: [buyer: The party that pays]]"


def test_convert_rejects_non_object_root() -> None:
    with pytest.raises(ValueError):
        convert_json_schema({"type": "array", "items": {"type": "string"}})


def test_conversion_hash_tracks_fields_and_engine() -> None:
    conversion = convert_json_schema(SCHEMA)
    base = conversion_hash(conversion, ai_engine="GCP_AI_2", ai_instructions=None)
    assert base == conversion_hash(
        convert_json_schema(json.loads(json.dumps(SCHEMA))), ai_engine="GCP_AI_2", ai_instructions=None
    )
    assert base != conversion_hash(conversion, ai_engine="GCP_AI_3_TXT", ai_instructions=None)
    assert base != conversion_hash(conversion, ai_engine="GCP_AI_2", ai_instructions="be careful")


# ----------------------------------------------------------------------------
# Parseur result -> schema-shaped JSON
# ----------------------------------------------------------------------------


def test_decode_result_undoes_only_provider_encodings() -> None:
    conversion = convert_json_schema(SCHEMA)
    result = {
        "invoice_number": "INV-001",
        "invoice_date": "2024-03-01",
        "total": 1234.5,
        "page_count": 3.0,
        "paid": "true",
        "currency": "EUR",
        "priority": "2",
        "customer.id": "",
        "customer.address.street": "1 Main St",
        "customer.address.zip": None,
        "tags": [{"value": "urgent"}, {"value": "paper"}],
        "line_items": [
            {
                "description": "Widget",
                "quantity": 2.0,
                "unit_price": 10.25,
                "taxable": "false",
                "dimensions.width": 1.5,
                "dimensions.height": "",
                "discounts": [{"code": "SAVE10"}, {"code": "VIP"}],
            }
        ],
        # Metadata or unknown keys are ignored, never surfaced.
        "Document ID": "abc",
    }

    assert decode_result(result, conversion) == {
        "invoice_number": "INV-001",
        "invoice_date": "2024-03-01",
        "total": 1234.5,
        # NUMBER values pass through as the JSON numbers Parseur returns.
        "page_count": 3.0,
        "paid": True,
        "currency": "EUR",
        "priority": 2,
        "customer": {"id": None, "address": {"street": "1 Main St", "zip": None}},
        "tags": ["urgent", "paper"],
        "line_items": [
            {
                "description": "Widget",
                "quantity": 2.0,
                "unit_price": 10.25,
                "taxable": False,
                "dimensions": {"width": 1.5, "height": None},
                "discounts": [{"code": "SAVE10"}, {"code": "VIP"}],
            }
        ],
    }


def test_decode_result_omits_top_level_fields_parseur_did_not_return() -> None:
    assert decode_result({"total": 1}, convert_json_schema(SCHEMA)) == {"total": 1}


def test_decode_result_gives_every_row_every_column() -> None:
    """A row carries every column the mailbox defines, empty ones as explicit nulls, so all rows share one shape."""
    decoded = decode_result({"line_items": [{"description": "a"}, {"quantity": 2}]}, convert_json_schema(SCHEMA))
    assert decoded["line_items"][0] == {
        "description": "a",
        "quantity": None,
        "unit_price": None,
        "taxable": None,
        "dimensions": {"width": None, "height": None},
        "discounts": None,
    }
    assert sorted(decoded["line_items"][1]) == sorted(decoded["line_items"][0])


def test_decode_result_reads_a_null_table_as_null_at_any_depth() -> None:
    decoded = decode_result(
        {"tags": None, "line_items": [{"description": "W", "discounts": None}]}, convert_json_schema(SCHEMA)
    )
    assert decoded["tags"] is None
    assert decoded["line_items"][0]["discounts"] is None


def test_decode_result_reads_nested_tables() -> None:
    """A table in a table in a table row (3 levels) decodes in place."""
    decoded = decode_result(
        {
            "tables": [
                {
                    "table_number": "1",
                    "column_headers": [{"value": "H1"}],
                    "rows": [{"row_label": "Total", "values": [{"value": "123"}, {"value": "456"}]}],
                }
            ]
        },
        convert_json_schema(DEEP_SCHEMA),
    )
    assert decoded["tables"] == [
        {
            "table_number": "1",
            "column_headers": ["H1"],
            "rows": [{"row_label": "Total", "values": ["123", "456"]}],
        }
    ]


# ----------------------------------------------------------------------------
# Provider flow with a fake HTTP session
# ----------------------------------------------------------------------------

RESULT = {
    "total": 99.0,
    "paid": "false",
    "tags": [{"value": "a"}],
    # With expand_result, dotted names come back as nested objects.
    "customer": {"id": "C1", "address": {"zip": "02903"}},
    "line_items": [{"description": "W", "dimensions": {"width": 2.0}}],
}


class _FakeResponse:
    def __init__(self, status_code: int, payload: Any = None):
        self.status_code = status_code
        self._payload = payload
        self.content = b"" if payload is None else json.dumps(payload).encode()
        self.text = self.content.decode()

    def json(self) -> Any:
        return self._payload


def _document(status: str = "PARSEDOK") -> _FakeResponse:
    # The API serializes the result as a JSON string.
    payload = {"id": "doc1", "status": status, "credits_used": 3, "content": "<html>big</html>"}
    return _FakeResponse(200, {**payload, "result": json.dumps(RESULT)})


POLL = ("GET", "/document/doc1")
UPLOAD = ("POST", "/parser/42/upload")


class _FakeSession:
    """Scripted Parseur API: list -> create -> upload -> poll (parsed).

    ``script[(method, path)]`` holds responses (or exceptions to raise) served first, in order, for that route.
    ``on_request`` runs before each response, e.g. to cancel the attempt mid-polling.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.headers: dict[str, str] = {}
        self.created_payloads: list[dict[str, Any]] = []
        self.listed: list[dict[str, Any]] = [{"id": 7, "name": "ExtractBench dead00"}]
        self.script: dict[tuple[str, str], list[Any]] = {}
        self.on_request: Any = None
        self.created_at: datetime | None = None

    def mount(self, prefix: str, adapter: Any) -> None:
        pass

    def request(self, method: str, url: str, **kwargs: Any) -> _FakeResponse:
        assert url.startswith(API_BASE_URL)
        path = url[len(API_BASE_URL) :]
        self.calls.append((method, path))
        if self.on_request is not None:
            self.on_request(method, path)
        if self.script.get((method, path)):
            response = self.script[(method, path)].pop(0)
            if isinstance(response, Exception):
                raise response
            return response
        if method == "GET" and path == "/parser":
            return _FakeResponse(200, {"results": self.listed, "current": 1, "total": 1})
        if method == "POST" and path == "/parser":
            self.created_payloads.append(kwargs["json"])
            self.created_at = datetime.now()
            return _FakeResponse(201, {"id": 42, "name": kwargs["json"]["name"]})
        if method == "POST" and path.endswith("/upload"):
            return _FakeResponse(201, {"message": "OK", "attachments": [{"name": "doc.pdf", "DocumentID": "doc1"}]})
        if (method, path) == POLL:
            return _document()
        return _FakeResponse(404, {"detail": f"unexpected {method} {path}"})


def _make_provider(monkeypatch: pytest.MonkeyPatch, **config: Any) -> tuple[ParseurProvider, _FakeSession]:
    monkeypatch.setenv("PARSEUR_API_KEY", "k")
    config = {"price_per_page_usd": 0.05, "poll_interval_s": 0, "max_requests_per_second": 0, **config}
    prov = ParseurProvider("parseur", config)
    fake = _FakeSession()
    prov._session = fake  # type: ignore[assignment]
    return prov, fake


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> tuple[ParseurProvider, _FakeSession]:
    return _make_provider(monkeypatch)


def _request(tmp_path: Path, schema: dict[str, Any] = SCHEMA) -> tuple[PipelineSpec, InferenceRequest]:
    pdf = tmp_path / "doc.pdf"
    writer = PdfWriter()
    for _ in range(2):
        writer.add_blank_page(width=612, height=792)
    with pdf.open("wb") as handle:
        writer.write(handle)
    pipeline = PipelineSpec(
        pipeline_name="parseur_test", provider_name="parseur", product_type=ProductType.EXTRACT, config={}
    )
    request = InferenceRequest(
        example_id="ex1", source_file_path=str(pdf), product_type=ProductType.EXTRACT, schema_override=schema
    )
    return pipeline, request


def _uploads(fake: _FakeSession) -> int:
    return sum(1 for method, path in fake.calls if method == "POST" and path.endswith("/upload"))


def test_registered_pipeline() -> None:
    pipeline = get_pipeline("parseur_extract_v3_vision")
    assert pipeline.provider_name == "parseur"
    assert pipeline.config == {"ai_engine": "GCP_AI_2", "price_per_page_usd": 0.05}
    assert pipeline.per_file_timeout is None


def test_run_inference_creates_pooled_mailbox_and_polls(
    provider: tuple[ParseurProvider, _FakeSession], tmp_path: Path
) -> None:
    prov, fake = provider
    # Ingestion is asynchronous: the document is not visible right after upload.
    fake.script[POLL] = [_FakeResponse(404), _FakeResponse(200, {"id": "doc1", "status": "PROGRESS"})]
    pipeline, request = _request(tmp_path)

    raw = prov.run_inference(pipeline, request)

    assert [c for c in fake.calls if c[0] == "POST"] == [("POST", "/parser"), UPLOAD]
    assert fake.calls.count(POLL) == 3
    payload = fake.created_payloads[0]
    assert payload["ai_engine"] == "GCP_AI_2"
    assert "ai_instructions" not in payload
    assert payload["expand_result"] is True
    assert payload["name"].startswith("ExtractBench ")
    assert payload["parser_object_set"] == convert_json_schema(SCHEMA)["parser_object_set"]
    assert raw.raw_output["mailbox_id"] == 42
    # The latency clock starts at the upload, after the one-off mailbox setup.
    assert fake.created_at is not None and raw.started_at >= fake.created_at
    assert raw.raw_output["document"]["result"]["total"] == 99.0
    assert "content" not in raw.raw_output["document"]

    normalized = prov.normalize(raw)
    assert normalized.output.extracted_data == {
        "total": 99.0,
        "paid": False,
        "tags": ["a"],
        "customer": {"id": "C1", "address": {"zip": "02903"}},
        "line_items": [
            {
                "description": "W",
                "quantity": None,
                "unit_price": None,
                "taxable": None,
                "dimensions": {"width": 2.0, "height": None},
                "discounts": None,
            }
        ],
    }
    assert normalized.output.field_citations == []
    assert normalized.raw_output["cost_usd"] == pytest.approx(0.10)
    assert normalized.raw_output["credits_used"] == 3.0


def test_normalize_after_a_json_round_trip(provider: tuple[ParseurProvider, _FakeSession], tmp_path: Path) -> None:
    """Re-normalizing a saved raw result gives the same output."""
    prov, _ = provider
    raw = prov.run_inference(*_request(tmp_path))
    reloaded = type(raw).model_validate_json(raw.model_dump_json())
    assert prov.normalize(reloaded).output == prov.normalize(raw).output


def test_pooled_mailboxes_are_listed_once_and_reused(
    provider: tuple[ParseurProvider, _FakeSession], tmp_path: Path
) -> None:
    """A mailbox already in the account, on any listing page, is reused by every document of its schema."""
    prov, fake = provider
    pipeline, request = _request(tmp_path)
    schema_hash = conversion_hash(convert_json_schema(SCHEMA), ai_engine="GCP_AI_2", ai_instructions="")
    fake.script[("GET", "/parser")] = [
        _FakeResponse(200, {"results": fake.listed, "current": 1, "total": 2}),
        _FakeResponse(200, {"results": [{"id": 9, "name": f"ExtractBench {schema_hash}"}], "current": 2, "total": 2}),
    ]

    for _ in range(2):
        assert prov.run_inference(pipeline, request).raw_output["mailbox_id"] == 9
    assert fake.created_payloads == []
    assert [c for c in fake.calls if c != POLL] == [
        ("GET", "/parser"),
        ("GET", "/parser"),
        ("POST", "/parser/9/upload"),
        ("POST", "/parser/9/upload"),
    ]


def test_mailbox_creation_error_lists_again_on_retry(
    provider: tuple[ParseurProvider, _FakeSession], tmp_path: Path
) -> None:
    """The server may have created the mailbox before a 5xx: the retry lists the account again."""
    prov, fake = provider
    pipeline, request = _request(tmp_path)
    fake.script[("POST", "/parser")] = [_FakeResponse(503)]
    with pytest.raises(ProviderTransientError):
        prov.run_inference(pipeline, request)
    prov.run_inference(pipeline, request)
    assert fake.calls.count(("GET", "/parser")) == 2


def test_mailbox_instructions_carry_the_schema_description(
    provider: tuple[ParseurProvider, _FakeSession], tmp_path: Path
) -> None:
    """The schema's own description is the document-level instruction: it goes to the mailbox AI instructions."""
    prov, fake = provider
    described = {**SCHEMA, "description": "  Use the figures of the most recent period.  "}
    pipeline, request = _request(tmp_path, described)
    prov.run_inference(pipeline, request)
    assert fake.created_payloads[0]["ai_instructions"] == "Use the figures of the most recent period."
    # The instructions are part of the mailbox identity: the plain schema gets its own mailbox.
    prov.run_inference(pipeline, request.model_copy(update={"schema_override": SCHEMA}))
    assert len(fake.created_payloads) == 2


@pytest.mark.parametrize("status", ["PARSEDOK", "EXPORTKO"])
def test_run_inference_accepts_a_parsed_document(
    provider: tuple[ParseurProvider, _FakeSession], tmp_path: Path, status: str
) -> None:
    """EXPORTKO means only an export of the account failed: the extraction itself is good."""
    prov, fake = provider
    fake.script[POLL] = [_document(status)]
    assert prov.run_inference(*_request(tmp_path)).raw_output["document"]["status"] == status


@pytest.mark.parametrize(("status", "message"), [("PARSEDKO", "PARSEDKO"), ("QUOTAEXC", "out of credits")])
def test_run_inference_fails_on_a_failed_document(
    provider: tuple[ParseurProvider, _FakeSession], tmp_path: Path, status: str, message: str
) -> None:
    prov, fake = provider
    fake.script[POLL] = [_FakeResponse(200, {"id": "doc1", "status": status})]
    with pytest.raises(ProviderPermanentError, match=message):
        prov.run_inference(*_request(tmp_path))


@pytest.mark.parametrize(
    ("poll", "last_status"),
    [(_FakeResponse(404), "not found"), (_FakeResponse(200, {"status": "PROGRESS"}), "PROGRESS")],
)
def test_run_inference_gives_up_at_the_deadline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, poll: _FakeResponse, last_status: str
) -> None:
    prov, fake = _make_provider(monkeypatch, timeout_s=0)
    fake.script[POLL] = [poll]
    with pytest.raises(ProviderPermanentError, match=f"still '{last_status}'"):
        prov.run_inference(*_request(tmp_path))


def test_run_inference_requires_schema(provider: tuple[ParseurProvider, _FakeSession], tmp_path: Path) -> None:
    prov, _ = provider
    pipeline, request = _request(tmp_path)
    with pytest.raises(ProviderPermanentError, match="schema_override"):
        prov.run_inference(pipeline, request.model_copy(update={"schema_override": None}))


def test_polling_rides_out_transient_errors(provider: tuple[ParseurProvider, _FakeSession], tmp_path: Path) -> None:
    prov, fake = provider
    fake.script[POLL] = [
        _FakeResponse(502),
        requests.exceptions.ChunkedEncodingError("connection broken"),
        requests.ConnectionError("reset"),
        _document(),
    ]
    assert prov.run_inference(*_request(tmp_path)).raw_output["document_id"] == "doc1"


def test_polling_errors_in_a_row_are_permanent(provider: tuple[ParseurProvider, _FakeSession], tmp_path: Path) -> None:
    prov, fake = provider
    fake.script[POLL] = [_FakeResponse(503)] * MAX_POLL_ERRORS
    with pytest.raises(ProviderPermanentError, match="polling errors in a row") as polling:
        prov.run_inference(*_request(tmp_path))
    assert polling.value.job_id == "doc1"
    assert _uploads(fake) == 1


def test_rate_limited_polls_do_not_count_as_errors(
    provider: tuple[ParseurProvider, _FakeSession], tmp_path: Path
) -> None:
    prov, fake = provider
    fake.script[POLL] = [_FakeResponse(429)] * (MAX_POLL_ERRORS + 2) + [_document()]
    assert prov.run_inference(*_request(tmp_path)).raw_output["document_id"] == "doc1"


def test_parsedok_before_its_result_is_polled_again(
    provider: tuple[ParseurProvider, _FakeSession], tmp_path: Path
) -> None:
    """A document can read PARSEDOK a moment before its result is available: the poll then sees no result yet."""
    prov, fake = provider
    early = _FakeResponse(200, {"id": "doc1", "status": "PARSEDOK", "result": None, "credits_used": 0})
    fake.script[POLL] = [early, early, _document()]
    result = prov.run_inference(*_request(tmp_path))
    assert result.raw_output["document"]["credits_used"] == 3
    assert fake.calls.count(POLL) == 3


def test_parsedok_without_result_fails_after_a_few_polls(
    provider: tuple[ParseurProvider, _FakeSession], tmp_path: Path
) -> None:
    prov, fake = provider
    early = _FakeResponse(200, {"id": "doc1", "status": "PARSEDOK", "result": None})
    fake.script[POLL] = [early] * (MAX_EMPTY_RESULT_POLLS + 1)
    with pytest.raises(ProviderPermanentError, match="is PARSEDOK but has no result"):
        prov.run_inference(*_request(tmp_path))
    assert fake.calls.count(POLL) == MAX_EMPTY_RESULT_POLLS + 1


@pytest.mark.parametrize("error", [_FakeResponse(504), requests.ConnectionError("reset"), requests.Timeout("slow")])
def test_failed_upload_is_permanent(provider: tuple[ParseurProvider, _FakeSession], tmp_path: Path, error: Any) -> None:
    """The server may have stored the file before the error: a runner retry would bill it again."""
    prov, fake = provider
    fake.script[UPLOAD] = [error]
    with pytest.raises(ProviderPermanentError, match="Upload of doc.pdf failed"):
        prov.run_inference(*_request(tmp_path))
    assert _uploads(fake) == 1


def test_rate_limited_upload_is_left_to_the_runner(
    provider: tuple[ParseurProvider, _FakeSession], tmp_path: Path
) -> None:
    """HTTP 429 is retryable, an upload included: the server rejected the request outright."""
    prov, fake = provider
    fake.script[UPLOAD] = [_FakeResponse(429, {"detail": "slow down"})]
    with pytest.raises(ProviderRateLimitError):
        prov.run_inference(*_request(tmp_path))


def test_runner_retry_after_cancel_polls_the_same_document(
    provider: tuple[ParseurProvider, _FakeSession], tmp_path: Path
) -> None:
    """A runner timeout cancels the attempt; its retry resumes polling instead of uploading and billing again."""
    prov, fake = provider
    pipeline, request = _request(tmp_path)
    fake.script[POLL] = [_FakeResponse(200, {"id": "doc1", "status": "PROGRESS"})] * 2
    fake.on_request = lambda method, path: (method, path) == POLL and prov.cancel("ex1")
    with pytest.raises(ProviderTransientError, match="stopped by the runner"):
        prov.run_inference(pipeline, request)
    first_calls = len(fake.calls)

    fake.on_request = None
    retried_at = datetime.now()
    raw = prov.run_inference(pipeline, request)
    assert _uploads(fake) == 1
    assert fake.calls[first_calls:] == [POLL, POLL]
    # Latency still counts from the upload, in the first attempt.
    assert raw.started_at < retried_at

    # Once the document is done, a new run of the same example uploads again.
    prov.run_inference(pipeline, request)
    assert _uploads(fake) == 2


def test_cancel_without_a_running_attempt(provider: tuple[ParseurProvider, _FakeSession]) -> None:
    prov, _ = provider
    assert prov.cancel("ex1") is False


def test_throttle_spaces_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PARSEUR_API_KEY", "k")
    prov = ParseurProvider("parseur", {"max_requests_per_second": 4})
    sleeps: list[float] = []
    monkeypatch.setattr("extract_bench.inference.providers.extract.parseur.time.monotonic", lambda: 100.0)
    monkeypatch.setattr("extract_bench.inference.providers.extract.parseur.time.sleep", sleeps.append)
    for _ in range(3):
        prov._throttle()
    # First call goes through immediately, the next two wait 0.25 s and 0.5 s.
    assert [round(s, 3) for s in sleeps] == [0.25, 0.5]
